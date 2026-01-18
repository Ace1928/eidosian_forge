import warnings
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from ..observer import _with_args, ObserverBase, PerChannelMinMaxObserver
from ..utils import _parent_name, check_min_max_valid
from .utils import (
def maybe_get_weight_eq_obs_node(op_node: Node, modules: Dict[str, nn.Module]) -> Optional[Node]:
    """ Gets the weight equalization observer node if it exists.
    """
    assert op_node.op == 'call_function'
    for node_arg in op_node.args:
        if node_arg_is_weight(op_node, node_arg):
            assert isinstance(node_arg, Node) and node_arg.op == 'call_module' and isinstance(modules[str(node_arg.target)], _WeightEqualizationObserver)
            return node_arg
    return None