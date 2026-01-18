import argparse
import itertools
import math
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging
def create_rename_keys_backbone(config):
    rename_keys = []
    rename_keys.append(('cls_token', 'backbone.embeddings.cls_token'))
    rename_keys.append(('mask_token', 'backbone.embeddings.mask_token'))
    rename_keys.append(('pos_embed', 'backbone.embeddings.position_embeddings'))
    rename_keys.append(('patch_embed.proj.weight', 'backbone.embeddings.patch_embeddings.projection.weight'))
    rename_keys.append(('patch_embed.proj.bias', 'backbone.embeddings.patch_embeddings.projection.bias'))
    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f'blocks.{i}.norm1.weight', f'backbone.encoder.layer.{i}.norm1.weight'))
        rename_keys.append((f'blocks.{i}.norm1.bias', f'backbone.encoder.layer.{i}.norm1.bias'))
        rename_keys.append((f'blocks.{i}.norm2.weight', f'backbone.encoder.layer.{i}.norm2.weight'))
        rename_keys.append((f'blocks.{i}.norm2.bias', f'backbone.encoder.layer.{i}.norm2.bias'))
        if config.backbone_config.use_swiglu_ffn:
            rename_keys.append((f'blocks.{i}.mlp.w12.weight', f'backbone.encoder.layer.{i}.mlp.w12.weight'))
            rename_keys.append((f'blocks.{i}.mlp.w12.bias', f'backbone.encoder.layer.{i}.mlp.w12.bias'))
            rename_keys.append((f'blocks.{i}.mlp.w3.weight', f'backbone.encoder.layer.{i}.mlp.w3.weight'))
            rename_keys.append((f'blocks.{i}.mlp.w3.bias', f'backbone.encoder.layer.{i}.mlp.w3.bias'))
        else:
            rename_keys.append((f'blocks.{i}.mlp.fc1.weight', f'backbone.encoder.layer.{i}.mlp.fc1.weight'))
            rename_keys.append((f'blocks.{i}.mlp.fc1.bias', f'backbone.encoder.layer.{i}.mlp.fc1.bias'))
            rename_keys.append((f'blocks.{i}.mlp.fc2.weight', f'backbone.encoder.layer.{i}.mlp.fc2.weight'))
            rename_keys.append((f'blocks.{i}.mlp.fc2.bias', f'backbone.encoder.layer.{i}.mlp.fc2.bias'))
        rename_keys.append((f'blocks.{i}.ls1.gamma', f'backbone.encoder.layer.{i}.layer_scale1.lambda1'))
        rename_keys.append((f'blocks.{i}.ls2.gamma', f'backbone.encoder.layer.{i}.layer_scale2.lambda1'))
        rename_keys.append((f'blocks.{i}.attn.proj.weight', f'backbone.encoder.layer.{i}.attention.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.attn.proj.bias', f'backbone.encoder.layer.{i}.attention.output.dense.bias'))
    rename_keys.append(('norm.weight', 'backbone.layernorm.weight'))
    rename_keys.append(('norm.bias', 'backbone.layernorm.bias'))
    return rename_keys