from typing import Callable, Dict, List, Set
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec
def init_batch_dim_size(self, batch_dim_size: int) -> None:
    """Initialize batch dim size base on the first input batch size."""
    if self.batch_dim_size != -1 and self.batch_dim_size != batch_dim_size:
        raise RuntimeError(f'batch dim size is already initialized! Found new batch size: {batch_dim_size} not matching existing batch dim size: {self.batch_dim_size}!')
    self.batch_dim_size = batch_dim_size