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
def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict, is_swin: bool):
    dst_prefix: str = 'pixel_level_module.decoder'
    src_prefix: str = 'sem_seg_head.pixel_decoder'
    if is_swin:
        self.replace_swin_backbone(dst_state_dict, src_state_dict, self.config)
    else:
        self.replace_dinat_backbone(dst_state_dict, src_state_dict, self.config)

    def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
        return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

    def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
        self_attn_keys = []
        self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.attention_weights', f'{dst_prefix}.attention_weights'))
        self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.output_proj', f'{dst_prefix}.output_proj'))
        self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.sampling_offsets', f'{dst_prefix}.sampling_offsets'))
        self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.value_proj', f'{dst_prefix}.value_proj'))
        return self_attn_keys

    def rename_keys_for_encoder_layer(src_prefix: str, dst_prefix: str):
        encoder_keys = []
        encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.fc1'))
        encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.fc2'))
        encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm1', f'{dst_prefix}.self_attn_layer_norm'))
        encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm2', f'{dst_prefix}.final_layer_norm'))
        encoder_keys.extend(rename_keys_for_self_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
        return encoder_keys
    renamed_keys = [(f'{src_prefix}.adapter_1.weight', f'{dst_prefix}.adapter_1.0.weight'), (f'{src_prefix}.adapter_1.norm.weight', f'{dst_prefix}.adapter_1.1.weight'), (f'{src_prefix}.adapter_1.norm.bias', f'{dst_prefix}.adapter_1.1.bias')]
    renamed_keys.extend([(f'{src_prefix}.layer_1.weight', f'{dst_prefix}.layer_1.0.weight'), (f'{src_prefix}.layer_1.norm.weight', f'{dst_prefix}.layer_1.1.weight'), (f'{src_prefix}.layer_1.norm.bias', f'{dst_prefix}.layer_1.1.bias')])
    for i in range(3):
        for j in range(2):
            renamed_keys.extend([(f'{src_prefix}.input_proj.{i}.{j}.weight', f'{dst_prefix}.input_projections.{i}.{j}.weight'), (f'{src_prefix}.input_proj.{i}.{j}.bias', f'{dst_prefix}.input_projections.{i}.{j}.bias')])
    renamed_keys.extend([(f'{src_prefix}.transformer.level_embed', f'{dst_prefix}.level_embed')])
    for layer_idx in range(self.config.encoder_layers):
        renamed_keys.extend(rename_keys_for_encoder_layer(f'{src_prefix}.transformer.encoder.layers.{layer_idx}', f'{dst_prefix}.encoder.layers.{layer_idx}'))
    renamed_keys.extend([(f'{src_prefix}.mask_features.weight', f'{dst_prefix}.mask_projection.weight'), (f'{src_prefix}.mask_features.bias', f'{dst_prefix}.mask_projection.bias')])
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)