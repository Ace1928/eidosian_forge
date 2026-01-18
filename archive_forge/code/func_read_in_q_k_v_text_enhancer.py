import argparse
import requests
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import (
def read_in_q_k_v_text_enhancer(state_dict, config):
    hidden_size = config.hidden_size
    for idx in range(config.encoder_layers):
        in_proj_weight = state_dict.pop(f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.in_proj_weight')
        in_proj_bias = state_dict.pop(f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.in_proj_bias')
        state_dict[f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.query.weight'] = in_proj_weight[:hidden_size, :]
        state_dict[f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.query.bias'] = in_proj_bias[:hidden_size]
        state_dict[f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.key.weight'] = in_proj_weight[hidden_size:hidden_size * 2, :]
        state_dict[f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.key.bias'] = in_proj_bias[hidden_size:hidden_size * 2]
        state_dict[f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.value.weight'] = in_proj_weight[-hidden_size:, :]
        state_dict[f'model.encoder.layers.{idx}.text_enhancer_layer.self_attn.value.bias'] = in_proj_bias[-hidden_size:]