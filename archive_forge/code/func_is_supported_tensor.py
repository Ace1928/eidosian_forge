import itertools
import sys
from functools import wraps
from typing import (
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import (
from torch.distributed._tensor import (
from torch.distributed._tensor.placement_types import Placement
def is_supported_tensor(self, t: torch.Tensor) -> bool:
    return not any([t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized, t.is_nested, torch._is_functional_tensor(t), t.is_neg(), t.is_conj(), t.device.type in ('lazy', 'meta')])