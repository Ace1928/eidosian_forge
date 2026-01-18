import enum
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import (
from .configuration_tapas import TapasConfig
def gather(values, index, name='segmented_gather'):
    """
    Gathers from *values* using the index map. For each element in the domain of the index map this operation looks up
    a value for that index in *values*. Two elements from the same segment always get assigned the same value.

    Args:
        values (`torch.Tensor` of shape (B1, ..., Bn, num_segments, V1, ...)):
            Tensor with segment values.
        index (`IndexMap` of shape (B1, ..., Bn, I1, ..., Ik)):
            IndexMap.
        name (`str`, *optional*, defaults to 'segmented_gather'):
            Name for the operation. Currently not used

    Returns:
        `tuple(torch.Tensor)`: Tensor of shape (B1, ..., Bn, I1, ..., Ik, V1, ...) with the gathered values.
    """
    indices = index.indices
    if len(values.shape[index.batch_dims:]) < 2:
        return torch.gather(values, index.batch_dims, indices.view(values.size()[0], -1)).view(indices.size())
    else:
        indices = indices.unsqueeze(-1).expand(values.shape)
        return torch.gather(values, index.batch_dims, indices)