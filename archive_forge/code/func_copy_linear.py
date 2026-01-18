import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_linear(hf_linear, pt_weights, prefix):
    hf_linear.weight.data = pt_weights[f'{prefix}.weight'].data
    hf_linear.bias.data = pt_weights[f'{prefix}.bias'].data