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
def node_supports_equalization(node: Node, modules) -> bool:
    """ Checks if the current node supports equalization
    Currently we only support nn.Linear/F.Linear and nn.Conv/F.conv layers
    """
    if node.op == 'call_module':
        return nn_module_supports_equalization(modules[str(node.target)]) or fused_module_supports_equalization(modules[str(node.target)]) or custom_module_supports_equalization(modules[str(node.target)])
    elif node.op == 'call_function':
        return node.target in [F.linear, F.conv1d, F.conv2d, F.conv3d]
    return False