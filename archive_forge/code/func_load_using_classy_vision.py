import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.utils import logging
def load_using_classy_vision(checkpoint_url: str, model_func: Callable[[], nn.Module]) -> Tuple[nn.Module, Dict]:
    files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location='cpu')
    model = model_func()
    model_state_dict = files['classy_state_dict']['base_model']['model']
    state_dict = model_state_dict['trunk']
    model.load_state_dict(state_dict)
    return (model.eval(), model_state_dict['heads'])