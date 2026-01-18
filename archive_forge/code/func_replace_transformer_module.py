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
def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module'
    src_prefix: str = 'sem_seg_head.predictor'

    def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
        return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

    def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
        attn_keys = [(f'{src_prefix}.in_proj_bias', f'{dst_prefix}.in_proj_bias'), (f'{src_prefix}.in_proj_weight', f'{dst_prefix}.in_proj_weight')]
        attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.out_proj', f'{dst_prefix}.out_proj'))
        return attn_keys

    def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
        attn_keys = []
        attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.out_proj', f'{dst_prefix}.out_proj'))
        return attn_keys

    def rename_keys_for_query_transformer_layer(src_prefix: str, dst_prefix: str):
        query_transformer_layer_keys = []
        query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.linear1'))
        query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.linear2'))
        query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm1', f'{dst_prefix}.norm1'))
        query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm2', f'{dst_prefix}.norm2'))
        query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm3', f'{dst_prefix}.norm3'))
        query_transformer_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
        query_transformer_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.multihead_attn', f'{dst_prefix}.multihead_attn'))
        return query_transformer_layer_keys

    def rename_keys_for_cross_attn_layer(src_prefix: str, dst_prefix: str):
        cross_attn_layer_keys = []
        cross_attn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
        cross_attn_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.multihead_attn', f'{dst_prefix}.multihead_attn'))
        return cross_attn_layer_keys

    def rename_keys_for_self_attn_layer(src_prefix: str, dst_prefix: str):
        self_attn_layer_keys = []
        self_attn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
        self_attn_layer_keys.extend(rename_keys_for_self_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
        return self_attn_layer_keys

    def rename_keys_for_ffn_layer(src_prefix: str, dst_prefix: str):
        ffn_layer_keys = []
        ffn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.linear1'))
        ffn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.linear2'))
        ffn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
        return ffn_layer_keys

    def rename_keys_for_transformer_decoder_layer(src_prefix: str, dst_prefix: str, idx: int):
        transformer_decoder_layer_keys = []
        transformer_decoder_layer_keys.extend(rename_keys_for_cross_attn_layer(f'{src_prefix}.transformer_cross_attention_layers.{idx}', f'{dst_prefix}.{idx}.cross_attn'))
        transformer_decoder_layer_keys.extend(rename_keys_for_self_attn_layer(f'{src_prefix}.transformer_self_attention_layers.{idx}', f'{dst_prefix}.{idx}.self_attn'))
        transformer_decoder_layer_keys.extend(rename_keys_for_ffn_layer(f'{src_prefix}.transformer_ffn_layers.{idx}', f'{dst_prefix}.{idx}.ffn'))
        return transformer_decoder_layer_keys
    renamed_keys = [(f'{src_prefix}.query_embed.weight', f'{dst_prefix}.queries_embedder.weight'), (f'{src_prefix}.level_embed.weight', f'{dst_prefix}.level_embed.weight')]
    renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.decoder_norm', f'{dst_prefix}.decoder.decoder_norm'))
    renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.class_input_proj', f'{dst_prefix}.decoder.query_input_projection'))
    renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.class_embed', f'{dst_prefix}.decoder.class_embed'))
    for i in range(3):
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mask_embed.layers.{i}', f'{dst_prefix}.decoder.mask_embed.layers.{i}.0'))
    renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.class_transformer.decoder.norm', f'{dst_prefix}.decoder.query_transformer.decoder.norm'))
    for i in range(self.config.query_dec_layers):
        renamed_keys.extend(rename_keys_for_query_transformer_layer(f'{src_prefix}.class_transformer.decoder.layers.{i}', f'{dst_prefix}.decoder.query_transformer.decoder.layers.{i}'))
    for i in range(self.config.decoder_layers - 1):
        renamed_keys.extend(rename_keys_for_transformer_decoder_layer(f'{src_prefix}', f'{dst_prefix}.decoder.layers', i))
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)