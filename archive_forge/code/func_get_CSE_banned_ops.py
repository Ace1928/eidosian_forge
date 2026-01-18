from typing import Dict, Tuple, Any
import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten
from torch.fx import GraphModule, Graph
from torch.fx import Node
@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def get_CSE_banned_ops():
    return rand_ops.union(inplace_ops)