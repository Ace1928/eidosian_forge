import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import (
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ''
        else:
            prefix = 'vit.'
        in_proj_weight = state_dict.pop(f'blocks.{i}.attn.qkv.weight')
        in_proj_bias = state_dict.pop(f'blocks.{i}.attn.qkv.bias')
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.query.weight'] = in_proj_weight[:config.hidden_size, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.query.bias'] = in_proj_bias[:config.hidden_size]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.key.weight'] = in_proj_weight[config.hidden_size:config.hidden_size * 2, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.key.bias'] = in_proj_bias[config.hidden_size:config.hidden_size * 2]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.value.weight'] = in_proj_weight[-config.hidden_size:, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.value.bias'] = in_proj_bias[-config.hidden_size:]