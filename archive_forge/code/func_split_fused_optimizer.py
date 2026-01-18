import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def split_fused_optimizer(gm: IterGraphModule, optim_block: FusedOptimizerBlock, split_gradients: Set[fx.Node]) -> Tuple[FusedOptimizerBlock, FusedOptimizerBlock]:
    if not split_gradients:
        raise ValueError('The given split_gradients is empty.')
    if str(optim_block.optim.optim_node.target).startswith('aten._fused_adam'):
        return _split_fused_adam(gm, optim_block, split_gradients)
    else:
        raise NotImplementedError('Only fused_adam is supported now')