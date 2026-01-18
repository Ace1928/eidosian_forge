import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_layers(hf_layers, pt_weights, prefix):
    for layer_id, hf_layer in enumerate(hf_layers):
        copy_layer(hf_layer, pt_weights, f'{prefix}.{layer_id}')