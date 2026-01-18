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
def get_fused_optimizer_block(optim_node: fx.Node) -> FusedOptimizerBlock:
    """Given a fused optimizer node and return the FusedOptimizerBlock."""
    MAX_STEP_DISTANCE = 5
    nodes = collections.deque([optim_node, None])
    step_node = optim_node
    distance = 0
    while nodes and distance < MAX_STEP_DISTANCE:
        node = nodes.popleft()
        if node is None:
            distance += 1
            if nodes:
                nodes.append(None)
            continue
        elif node.op == OP.CALL_FUNCTION and str(node.target).startswith('aten._foreach_add'):
            step_node = node
            break
        else:
            nodes.extend((a for a in pytree.arg_tree_leaves(*node.args, **node.kwargs) if isinstance(a, fx.Node)))
    if step_node == optim_node:
        raise RuntimeError(f'Cannot find step node (foreach_add) for the optimizer node {optim_node} with {MAX_STEP_DISTANCE} BFS distance. The API design does not match the tracing graph.')
    step = ForeachAddBlock(step_node, generate_output=False)
    optim = FusedAdamBlock(optim_node, generate_output=False)
    return FusedOptimizerBlock(step, optim)