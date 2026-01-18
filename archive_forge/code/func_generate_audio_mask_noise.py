import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
def generate_audio_mask_noise(audio_values, audio_mask=None, mask_ratio=0.75, mask_type='patch-level', freq_len=8):
    """Generate noise for audio masking."""
    batch_size, seq_len = audio_values.shape[:2]
    if mask_type == 'frame-level':
        num_time_patches = seq_len // freq_len
        noise = torch.rand(batch_size, num_time_patches, device=audio_values.device).unsqueeze(-1).repeat(1, 1, freq_len).view(batch_size, seq_len)
    elif mask_type == 'patch-level':
        noise = torch.rand(batch_size, seq_len, device=audio_values.device)
    len_keep = int(seq_len * (1 - mask_ratio))
    return (noise, len_keep)