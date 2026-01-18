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
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from torch import Tensor, nn
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import (
from transformers.utils import logging
def rename_keys_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module.decoder'
    src_prefix: str = 'sem_seg_head.predictor.transformer.decoder'
    rename_keys = []
    for i in range(self.config.decoder_config.decoder_layers):
        rename_keys.append((f'{src_prefix}.layers.{i}.self_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.self_attn.out_proj.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.self_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.self_attn.out_proj.bias'))
        rename_keys.append((f'{src_prefix}.layers.{i}.multihead_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.encoder_attn.out_proj.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.multihead_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.encoder_attn.out_proj.bias'))
        rename_keys.append((f'{src_prefix}.layers.{i}.linear1.weight', f'{dst_prefix}.layers.{i}.fc1.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.linear1.bias', f'{dst_prefix}.layers.{i}.fc1.bias'))
        rename_keys.append((f'{src_prefix}.layers.{i}.linear2.weight', f'{dst_prefix}.layers.{i}.fc2.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.linear2.bias', f'{dst_prefix}.layers.{i}.fc2.bias'))
        rename_keys.append((f'{src_prefix}.layers.{i}.norm1.weight', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.norm1.bias', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.bias'))
        rename_keys.append((f'{src_prefix}.layers.{i}.norm2.weight', f'{dst_prefix}.layers.{i}.encoder_attn_layer_norm.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.norm2.bias', f'{dst_prefix}.layers.{i}.encoder_attn_layer_norm.bias'))
        rename_keys.append((f'{src_prefix}.layers.{i}.norm3.weight', f'{dst_prefix}.layers.{i}.final_layer_norm.weight'))
        rename_keys.append((f'{src_prefix}.layers.{i}.norm3.bias', f'{dst_prefix}.layers.{i}.final_layer_norm.bias'))
    return rename_keys