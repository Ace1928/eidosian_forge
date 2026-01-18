import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
def space_to_depth(frames: torch.Tensor, temporal_block_size: int=1, spatial_block_size: int=1) -> torch.Tensor:
    """
    Space to depth transform. Rearranges blocks of spatial data, into depth.

    This function assumes the channels to be first, but will place the channels last after transformation.

    Based on https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15.
    """
    if len(frames.shape) == 4:
        batch_size, num_channels, height, width = frames.shape
        frames = frames.view(batch_size, num_channels, height // spatial_block_size, spatial_block_size, width // spatial_block_size, spatial_block_size)
        frames = frames.permute(0, 2, 4, 3, 5, 1).contiguous()
        frames = frames.view(batch_size, height // spatial_block_size, width // spatial_block_size, spatial_block_size ** 2 * num_channels)
        return frames
    elif len(frames.shape) == 5:
        batch_size, time, num_channels, height, width = frames.shape
        frames = frames.view(batch_size, time // temporal_block_size, temporal_block_size, num_channels, height // spatial_block_size, spatial_block_size, width // spatial_block_size, spatial_block_size)
        frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        frames = frames.view(batch_size, time // temporal_block_size, height // spatial_block_size, width // spatial_block_size, temporal_block_size * spatial_block_size ** 2 * num_channels)
        return frames
    else:
        raise ValueError('Frames should be of rank 4 (batch, channels, height, width) or rank 5 (batch, time, channels, height, width)')