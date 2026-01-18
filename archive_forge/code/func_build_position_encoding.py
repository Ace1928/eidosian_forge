import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == 'sine':
        position_embedding = DetaSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == 'learned':
        position_embedding = DetaLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f'Not supported {config.position_embedding_type}')
    return position_embedding