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
def create_rename_keys(config, base_model=False):
    rename_keys = []
    rename_keys.append(('cls_token', 'vit.embeddings.cls_token'))
    rename_keys.append(('pos_embed', 'vit.embeddings.position_embeddings'))
    rename_keys.append(('patch_embed.proj.weight', 'vit.embeddings.patch_embeddings.projection.weight'))
    rename_keys.append(('patch_embed.proj.bias', 'vit.embeddings.patch_embeddings.projection.bias'))
    rename_keys.append(('patch_embed.backbone.stem.conv.weight', 'vit.embeddings.patch_embeddings.backbone.bit.embedder.convolution.weight'))
    rename_keys.append(('patch_embed.backbone.stem.norm.weight', 'vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.weight'))
    rename_keys.append(('patch_embed.backbone.stem.norm.bias', 'vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.bias'))
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv1.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv1.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.bias'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv2.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv2.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.bias'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv3.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv3.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.bias'))
        rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.conv.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.conv.weight'))
        rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.weight'))
        rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.bias'))
    for i in range(config.num_hidden_layers):
        rename_keys.append((f'blocks.{i}.norm1.weight', f'vit.encoder.layer.{i}.layernorm_before.weight'))
        rename_keys.append((f'blocks.{i}.norm1.bias', f'vit.encoder.layer.{i}.layernorm_before.bias'))
        rename_keys.append((f'blocks.{i}.attn.proj.weight', f'vit.encoder.layer.{i}.attention.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.attn.proj.bias', f'vit.encoder.layer.{i}.attention.output.dense.bias'))
        rename_keys.append((f'blocks.{i}.norm2.weight', f'vit.encoder.layer.{i}.layernorm_after.weight'))
        rename_keys.append((f'blocks.{i}.norm2.bias', f'vit.encoder.layer.{i}.layernorm_after.bias'))
        rename_keys.append((f'blocks.{i}.mlp.fc1.weight', f'vit.encoder.layer.{i}.intermediate.dense.weight'))
        rename_keys.append((f'blocks.{i}.mlp.fc1.bias', f'vit.encoder.layer.{i}.intermediate.dense.bias'))
        rename_keys.append((f'blocks.{i}.mlp.fc2.weight', f'vit.encoder.layer.{i}.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.mlp.fc2.bias', f'vit.encoder.layer.{i}.output.dense.bias'))
    if base_model:
        rename_keys.extend([('norm.weight', 'layernorm.weight'), ('norm.bias', 'layernorm.bias'), ('pre_logits.fc.weight', 'pooler.dense.weight'), ('pre_logits.fc.bias', 'pooler.dense.bias')])
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith('vit') else pair for pair in rename_keys]
    else:
        rename_keys.extend([('norm.weight', 'vit.layernorm.weight'), ('norm.bias', 'vit.layernorm.bias'), ('head.weight', 'classifier.weight'), ('head.bias', 'classifier.bias')])
    return rename_keys