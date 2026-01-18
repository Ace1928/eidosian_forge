import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_autoformer import AutoformerConfig
def get_lagged_subsequences(self, sequence: torch.Tensor, subsequences_length: int, shift: int=0) -> torch.Tensor:
    """
        Returns lagged subsequences of a given sequence. Returns a tensor of shape (batch_size, subsequences_length,
        feature_size, indices_length), containing lagged subsequences. Specifically, lagged[i, j, :, k] = sequence[i,
        -indices[k]-subsequences_length+j, :].

        Args:
            sequence (`torch.Tensor` or shape `(batch_size, context_length,
                feature_size)`): The sequence from which lagged subsequences should be extracted.
            subsequences_length (`int`):
                Length of the subsequences to be extracted.
            shift (`int`, *optional* defaults to 0):
                Shift the lags by this amount back in the time index.
        """
    indices = [lag - shift for lag in self.config.lags_sequence]
    sequence_length = sequence.shape[1]
    if max(indices) + subsequences_length > sequence_length:
        raise ValueError(f'lags cannot go further than history length, found lag {max(indices)} while history length is only {sequence_length}')
    lagged_values = []
    for lag_index in indices:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        lagged_values.append(sequence[:, begin_index:end_index, ...])
    return torch.stack(lagged_values, dim=-1)