import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging
def replace_dinat_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: OneFormerConfig):
    dst_prefix: str = 'pixel_level_module.encoder'
    src_prefix: str = 'backbone'

    def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
        return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]
    renamed_keys = rename_keys_for_weight_bias(f'{src_prefix}.patch_embed.norm', f'{dst_prefix}.embeddings.norm')
    for i in range(2):
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.patch_embed.proj.{i}', f'{dst_prefix}.embeddings.patch_embeddings.projection.{i}'))
    num_layers = len(config.backbone_config.depths)
    for layer_idx in range(num_layers):
        for block_idx in range(config.backbone_config.depths[layer_idx]):
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.norm1', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.layernorm_before'))
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.norm2', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.layernorm_after'))
            renamed_keys.extend([(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.rpb', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.rpb')])
            src_att_weight = src_state_dict[f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.weight']
            src_att_bias = src_state_dict[f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.bias']
            size = src_att_weight.shape[0]
            offset = size // 3
            dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.query.weight'] = src_att_weight[:offset, :]
            dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.query.bias'] = src_att_bias[:offset]
            dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.key.weight'] = src_att_weight[offset:offset * 2, :]
            dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.key.bias'] = src_att_bias[offset:offset * 2]
            dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.value.weight'] = src_att_weight[-offset:, :]
            dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.value.bias'] = src_att_bias[-offset:]
            src_state_dict.pop(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.weight')
            src_state_dict.pop(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.bias')
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.proj', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.output.dense'))
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.mlp.fc1', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.intermediate.dense'))
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.mlp.fc2', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.output.dense'))
        if layer_idx < num_layers - 1:
            renamed_keys.extend([(f'{src_prefix}.levels.{layer_idx}.downsample.reduction.weight', f'{dst_prefix}.encoder.levels.{layer_idx}.downsample.reduction.weight'), (f'{src_prefix}.levels.{layer_idx}.downsample.norm.weight', f'{dst_prefix}.encoder.levels.{layer_idx}.downsample.norm.weight'), (f'{src_prefix}.levels.{layer_idx}.downsample.norm.bias', f'{dst_prefix}.encoder.levels.{layer_idx}.downsample.norm.bias')])
        renamed_keys.extend([(f'{src_prefix}.norm{layer_idx}.weight', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.weight'), (f'{src_prefix}.norm{layer_idx}.bias', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.bias')])
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)