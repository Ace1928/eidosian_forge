import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from transformers import (
from transformers.models.mask2former.modeling_mask2former import (
from transformers.utils import logging
def replace_maskformer_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig):
    dst_prefix: str = 'pixel_level_module.encoder'
    src_prefix: str = 'backbone'
    renamed_keys = [(f'{src_prefix}.patch_embed.proj.weight', f'{dst_prefix}.model.embeddings.patch_embeddings.projection.weight'), (f'{src_prefix}.patch_embed.proj.bias', f'{dst_prefix}.model.embeddings.patch_embeddings.projection.bias'), (f'{src_prefix}.patch_embed.norm.weight', f'{dst_prefix}.model.embeddings.norm.weight'), (f'{src_prefix}.patch_embed.norm.bias', f'{dst_prefix}.model.embeddings.norm.bias')]
    num_layers = len(config.backbone_config.depths)
    for layer_idx in range(num_layers):
        for block_idx in range(config.backbone_config.depths[layer_idx]):
            renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table')])
            src_att_weight = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight']
            src_att_bias = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias']
            size = src_att_weight.shape[0]
            offset = size // 3
            dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight'] = src_att_weight[:offset, :]
            dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias'] = src_att_bias[:offset]
            dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight'] = src_att_weight[offset:offset * 2, :]
            dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias'] = src_att_bias[offset:offset * 2]
            dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight'] = src_att_weight[-offset:, :]
            dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias'] = src_att_bias[-offset:]
            src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight')
            src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias')
            renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias')])
            renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias')])
            renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias')])
            renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index')])
        if layer_idx < num_layers - 1:
            renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.downsample.reduction.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.reduction.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.bias')])
        renamed_keys.extend([(f'{src_prefix}.norm{layer_idx}.weight', f'{dst_prefix}.hidden_states_norms.{layer_idx}.weight'), (f'{src_prefix}.norm{layer_idx}.bias', f'{dst_prefix}.hidden_states_norms.{layer_idx}.bias')])
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)