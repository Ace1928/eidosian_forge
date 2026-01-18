import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
def loss_iou(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
    """
        Measure the intersection over union.
        """
    inter = torch.min(candidates_end_time, end_time) - torch.max(candidates_start_time, start_time)
    union = torch.max(candidates_end_time, end_time) - torch.min(candidates_start_time, start_time)
    iou = 1 - inter.clamp(min=0) / union
    return iou