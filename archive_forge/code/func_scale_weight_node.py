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
def scale_weight_node(node: Node, modules: Dict[str, nn.Module], equalization_scale: torch.Tensor, next_equalization_scale: Optional[torch.Tensor]) -> None:
    """ Scale the weights for input-weight equalization by multiplying the
    weight by 1/equalization_scale and next_equalization_scale

    Args:
        node: Current node whose weights we want to scale
        equalization_scale: Current node's calculated equalization scale
        next_equalization_scale: Next node's calculated equalization scale if
           the following node needs to be equalized, 1 otherwise
    """
    if equalization_scale is None:
        return
    if fused_module_supports_equalization(modules[str(node.target)]):
        op_module = modules[str(node.target)][0]
    else:
        op_module = modules[str(node.target)]
    assert nn_module_supports_equalization(op_module) or custom_module_supports_equalization(op_module)
    weight = op_module.weight
    assert isinstance(weight, torch.Tensor)
    equalization_scale_reshaped = reshape_scale(equalization_scale, 1, weight)
    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale_reshaped))
    if next_equalization_scale is None:
        op_module.weight = nn.Parameter(scaled_weight)
        return
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, weight)
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)
    op_module.weight = nn.Parameter(scaled_weight)
    bias = op_module.bias
    if bias is None:
        return
    assert isinstance(bias, torch.Tensor)
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, bias)
    scaled_bias = torch.mul(bias, next_equalization_scale_reshaped)
    op_module.bias = nn.Parameter(scaled_bias)