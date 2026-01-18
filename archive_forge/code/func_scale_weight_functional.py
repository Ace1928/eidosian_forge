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
def scale_weight_functional(op_node: Node, model: GraphModule, modules: Dict[str, nn.Module], equalization_scale: torch.Tensor, next_equalization_scale: Optional[torch.Tensor]) -> None:
    """ Scales the weight value for functional layers
    """
    if equalization_scale is None:
        return
    weight_eq_obs_node = maybe_get_weight_eq_obs_node(op_node, modules)
    if weight_eq_obs_node is None:
        return
    weight_quant_obs_node = weight_eq_obs_node.args[0]
    if weight_quant_obs_node is None:
        return
    assert isinstance(weight_quant_obs_node, Node) and isinstance(modules[str(weight_quant_obs_node.target)], ObserverBase)
    weight_node = weight_quant_obs_node.args[0]
    if weight_node is None:
        return
    assert isinstance(weight_node, Node) and weight_node.op == 'get_attr'
    weight_parent_name, weight_name = _parent_name(weight_node.target)
    weight = getattr(modules[weight_parent_name], weight_name)
    equalization_scale_reshaped = reshape_scale(equalization_scale, 1, weight)
    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale_reshaped))
    if next_equalization_scale is None:
        setattr(modules[weight_parent_name], weight_name, scaled_weight)
        return
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, scaled_weight)
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)
    setattr(modules[weight_parent_name], weight_name, scaled_weight)
    assert torch.allclose(model.get_buffer(str(weight_node.target)), scaled_weight)
    bias_node = None
    for node in op_node.args:
        if isinstance(node, Node) and node.op == 'get_attr' and ('bias' in node.name):
            bias_node = node
            break
    if bias_node is None:
        return
    bias_parent_name, bias_name = _parent_name(bias_node.target)
    bias = getattr(modules[bias_parent_name], bias_name)
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, bias)
    scaled_bias = torch.mul(bias, next_equalization_scale_reshaped)
    setattr(modules[bias_parent_name], bias_name, scaled_bias)