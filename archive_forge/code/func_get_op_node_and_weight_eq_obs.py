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
def get_op_node_and_weight_eq_obs(input_eq_obs_node: Node, model: GraphModule, modules: Dict[str, nn.Module]) -> Tuple[Optional[Node], Optional[_WeightEqualizationObserver]]:
    """ Gets the following weight equalization observer. There should always
    exist a weight equalization observer after an input equalization observer.

    Returns the operation node that follows the input equalization observer node
    and the weight equalization observer
    """
    op_node = None
    for user in input_eq_obs_node.users.keys():
        if node_supports_equalization(user, modules):
            op_node = user
            break
    assert op_node is not None
    if op_node.op == 'call_module':
        maybe_equalization_node_name_to_config = _get_observed_graph_module_attr(model, 'equalization_node_name_to_qconfig')
        assert maybe_equalization_node_name_to_config is not None
        equalization_node_name_to_qconfig: Dict[str, Any] = maybe_equalization_node_name_to_config
        assert equalization_node_name_to_qconfig.get(op_node.name, None) is not None
        weight_eq_obs = equalization_node_name_to_qconfig.get(op_node.name, None).weight()
        assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
        return (op_node, weight_eq_obs)
    elif op_node.op == 'call_function':
        weight_node = maybe_get_weight_eq_obs_node(op_node, modules)
        if weight_node is not None:
            weight_eq_obs = modules[str(weight_node.target)]
            assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
            return (op_node, weight_eq_obs)
    return (None, None)