import argparse
import torch
from transformers import ChineseCLIPConfig, ChineseCLIPModel
def copy_layer(hf_layer, pt_weights, prefix):
    copy_linear(hf_layer.layer_norm1, pt_weights, f'{prefix}.ln_1')
    copy_linear(hf_layer.layer_norm2, pt_weights, f'{prefix}.ln_2')
    copy_mlp(hf_layer.mlp, pt_weights, f'{prefix}.mlp')
    copy_attn_layer(hf_layer.self_attn, pt_weights, f'{prefix}.attn')