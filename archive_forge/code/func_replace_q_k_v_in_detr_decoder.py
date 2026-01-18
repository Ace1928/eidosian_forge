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
def replace_q_k_v_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module.decoder'
    src_prefix: str = 'sem_seg_head.predictor.transformer.decoder'
    for i in range(self.config.decoder_config.decoder_layers):
        in_proj_weight = src_state_dict.pop(f'{src_prefix}.layers.{i}.self_attn.in_proj_weight')
        in_proj_bias = src_state_dict.pop(f'{src_prefix}.layers.{i}.self_attn.in_proj_bias')
        dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.q_proj.weight'] = in_proj_weight[:256, :]
        dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.q_proj.bias'] = in_proj_bias[:256]
        dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.k_proj.weight'] = in_proj_weight[256:512, :]
        dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.k_proj.bias'] = in_proj_bias[256:512]
        dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.v_proj.weight'] = in_proj_weight[-256:, :]
        dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.v_proj.bias'] = in_proj_bias[-256:]
        in_proj_weight_cross_attn = src_state_dict.pop(f'{src_prefix}.layers.{i}.multihead_attn.in_proj_weight')
        in_proj_bias_cross_attn = src_state_dict.pop(f'{src_prefix}.layers.{i}.multihead_attn.in_proj_bias')
        dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.q_proj.weight'] = in_proj_weight_cross_attn[:256, :]
        dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.q_proj.bias'] = in_proj_bias_cross_attn[:256]
        dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight'] = in_proj_weight_cross_attn[256:512, :]
        dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias'] = in_proj_bias_cross_attn[256:512]
        dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight'] = in_proj_weight_cross_attn[-256:, :]
        dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias'] = in_proj_bias_cross_attn[-256:]