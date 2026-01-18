import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
def interpolate(hidden_states, ratio):
    """
    Interpolate data in time domain. This is used to compensate the resolution reduction in downsampling of a CNN.

    Args:
        hidden_states (`torch.FloatTensor` of shape (batch_size, time_length, classes_num)):
            Input hidden states
        ratio (`int`):
            The ratio of the length of the output to the length of the input.
    """
    batch_size, time_length, classes_num = hidden_states.shape
    upsampled = hidden_states[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_length * ratio, classes_num)
    return upsampled