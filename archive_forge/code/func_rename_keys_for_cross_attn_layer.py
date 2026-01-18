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
def rename_keys_for_cross_attn_layer(src_prefix: str, dst_prefix: str):
    cross_attn_layer_keys = []
    cross_attn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
    cross_attn_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.multihead_attn', f'{dst_prefix}.multihead_attn'))
    return cross_attn_layer_keys