import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
@classmethod
def pad_local_tokens(cls, hidden_states, attention_mask, block_size):
    pad_size = block_size // 2
    mask_min_value = torch.finfo(hidden_states.dtype).min
    padded_hidden_states = torch.nn.functional.pad(hidden_states, pad=(0, 0, pad_size, pad_size))
    padded_mask = torch.nn.functional.pad(attention_mask, pad=(pad_size, pad_size), value=mask_min_value)
    return (padded_hidden_states, padded_mask)