# CSCI-SHU 360 Machine Learning Final Project
> Author: Ricercar Guo & Yiling Cao

 This is the final project of CSCI-SHU 360 Machine Learning. Trying to train a diffusion model for scribble image generation.

## Dataset
* https://huggingface.co/datasets/quickdraw
* https://github.com/googlecreativelab/quickdraw-dataset
* https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap%3Btab=objects?prefix=&forceOnObjectsSortingFiltering=false

## Getting Started With HPC
### NYU Greene HPC
* [NYU Greene HPC Main Page](https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc/getting-and-renewing-an-account?authuser=0#h.p_ID_34)
* [Conda Environment on HPC - Singularity Overlays for Miniconda](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda?authuser=0#h.u46va8o5agd6)
* [Running Jupyter Notebook on HPC - HPC Open On Demand](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/open-ondemand-ood-with-condasingularity?authuser=0)
### Useful notes
* [NYU HPC Notes By Hammond](https://abstracted-crime-34a.notion.site/63aae4cc39904d11a5c744f480a42017?v=261a410e1fe24d0294ed744c21a41015)

## Online Resources
### Train an unconditioned image generation model
* https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training

### Fine tune with LoRA
* https://huggingface.co/docs/diffusers/main/en/training/lora

### Schedulers
* https://huggingface.co/docs/diffusers/v0.9.0/en/api/schedulers

### Model evaluation
* https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation

## File Structure
*Last update: 2023.5.12*
```
├── Final_Report
│   ├── evaluation_output
│   ├── inference_output
│   ├── main.tex
│   └── references.bib
├── ML_Proposal.pdf
├── ML_final_project_guideline.docx
├── Playground
│   ├── butterfly_example.ipynb
│   ├── disassemble_model.ipynb
│   ├── face_example.ipynb
│   ├── full_train_lora_scribble.py
│   ├── nielsr-CelebA-faces
│   ├── scaddpm-butterflies-128
│   ├── scribble_example.ipynb
│   ├── sddata
│   ├── slurm-32426471.out
│   ├── train_lora.slurm
│   ├── train_text_to_image_lora.py
│   ├── train_text_to_image_lora_scribble.ipynb
│   ├── try_loading_datasets.ipynb
│   └── try_ndjson.ipynb
├── README.md
└── Training
    ├── categories.txt
    ├── dataset_example.ipynb
    ├── lora_output
    ├── mini_classes.txt
    ├── train_lora_scribble.py
    ├── train_lora_scribble.slurm
    ├── train_result_evaluation.ipynb
    └── train_result_inference.ipynb
```

