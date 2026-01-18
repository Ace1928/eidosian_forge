import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def get_valid_ratio(self, mask, dtype=torch.float32):
    """Get the valid ratio of all feature maps."""
    _, height, width = mask.shape
    valid_height = torch.sum(~mask[:, :, 0], 1)
    valid_width = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_heigth = valid_height.to(dtype) / height
    valid_ratio_width = valid_width.to(dtype) / width
    valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
    return valid_ratio