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
def loss_duration(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
    """
        Measure the difference of duration.
        """
    duration_candidates = torch.sub(candidates_end_time, candidates_start_time)
    duration_groundtruth = torch.sub(end_time, start_time)
    duration_diff = torch.square(torch.div(torch.sub(duration_candidates, duration_groundtruth), duration))
    duration_diff = duration_diff.clamp(min=0.4)
    return duration_diff