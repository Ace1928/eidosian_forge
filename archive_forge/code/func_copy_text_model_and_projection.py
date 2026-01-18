import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_text_model_and_projection(hf_model, pt_weights):
    hf_model.text_projection.weight.data = pt_weights['text_projection'].data.T
    for name, param in hf_model.text_model.named_parameters():
        param.data = pt_weights[f'bert.{name}'].data