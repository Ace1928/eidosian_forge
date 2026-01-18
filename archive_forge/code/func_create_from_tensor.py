from dataclasses import dataclass, field
from enum import Enum
from typing import List
import torch
from torch.distributed._shard.metadata import ShardMetadata
@staticmethod
def create_from_tensor(tensor: torch.Tensor) -> 'TensorProperties':
    return TensorProperties(dtype=tensor.dtype, layout=tensor.layout, requires_grad=tensor.requires_grad, memory_format=torch.contiguous_format, pin_memory=tensor.is_pinned())