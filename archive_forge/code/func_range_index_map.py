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
def range_index_map(batch_shape, num_segments, name='range_index_map'):
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (`torch.Size`):
            Batch shape
        num_segments (`int`):
            Number of segments
        name (`str`, *optional*, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(batch_shape, dtype=torch.long)
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)
    assert len(num_segments.size()) == 0
    indices = torch.arange(start=0, end=num_segments, device=num_segments.device)
    new_tensor = torch.cat([torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)], dim=0)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)
    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])