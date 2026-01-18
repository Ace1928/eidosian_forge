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
def clear_weight_quant_obs_node(op_node: Node, modules: Dict[str, nn.Module]) -> None:
    """ Given the operation node, we want find the corresponding quantization
    observer and reset its min/max values
    """
    weight_eq_obs_node = maybe_get_weight_eq_obs_node(op_node, modules)
    if weight_eq_obs_node is None:
        return
    weight_quant_obs_node = weight_eq_obs_node.args[0]
    if weight_quant_obs_node is None:
        return
    assert isinstance(weight_quant_obs_node, Node)
    weight_quant_obs = modules[str(weight_quant_obs_node.target)]
    assert isinstance(modules[str(weight_quant_obs_node.target)], ObserverBase)
    weight_quant_obs.reset_min_max_vals()