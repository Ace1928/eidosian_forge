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
def flatten(index, name='segmented_flatten'):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    *num_segments* * (k - 1). The result is a tensor with *num_segments* multiplied by the number of elements in the
    batch.

    Args:
        index (`IndexMap`):
            IndexMap to flatten.
        name (`str`, *optional*, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): The flattened IndexMap.
    """
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):
        offset = offset.unsqueeze(-1)
    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)