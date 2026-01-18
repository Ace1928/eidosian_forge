from typing import Dict, Tuple, Any
import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten
from torch.fx import GraphModule, Graph
from torch.fx import Node
def get_aten_target(node):
    if hasattr(node.target, 'overloadpacket'):
        return node.target.overloadpacket
    return node.target