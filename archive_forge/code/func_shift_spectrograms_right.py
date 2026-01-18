import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
def shift_spectrograms_right(input_values: torch.Tensor, reduction_factor: int=1):
    """
    Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
    """
    if reduction_factor > 1:
        input_values = input_values[:, reduction_factor - 1::reduction_factor]
    shifted_input_values = input_values.new_zeros(input_values.shape)
    shifted_input_values[:, 1:] = input_values[:, :-1].clone()
    shifted_input_values.masked_fill_(shifted_input_values == -100.0, 0.0)
    return shifted_input_values