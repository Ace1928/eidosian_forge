import argparse
import json
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from huggingface_hub import cached_download, hf_hub_download
from torch import Tensor
from transformers import AutoImageProcessor, VanConfig, VanForImageClassification
from transformers.models.deprecated.van.modeling_van import VanLayerScaling
from transformers.utils import logging
def copy_parameters(from_model: nn.Module, our_model: nn.Module) -> nn.Module:
    from_state_dict = from_model.state_dict()
    our_state_dict = our_model.state_dict()
    config = our_model.config
    all_keys = []
    for stage_idx in range(len(config.hidden_sizes)):
        for block_id in range(config.depths[stage_idx]):
            from_key = f'block{stage_idx + 1}.{block_id}.layer_scale_1'
            to_key = f'van.encoder.stages.{stage_idx}.layers.{block_id}.attention_scaling.weight'
            all_keys.append((from_key, to_key))
            from_key = f'block{stage_idx + 1}.{block_id}.layer_scale_2'
            to_key = f'van.encoder.stages.{stage_idx}.layers.{block_id}.mlp_scaling.weight'
            all_keys.append((from_key, to_key))
    for from_key, to_key in all_keys:
        our_state_dict[to_key] = from_state_dict.pop(from_key)
    our_model.load_state_dict(our_state_dict)
    return our_model