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
def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
    mesh_size = self.mesh.size()
    sharding_choices: List[Placement] = [Replicate()]
    if arg.dtype != torch.bool:
        sharding_choices = sharding_choices + [Shard(i) for i, s in enumerate(arg.shape) if s > 1 and s % mesh_size == 0]
    return sharding_choices