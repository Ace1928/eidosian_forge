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
def scale_input_observer(node: Node, modules: Dict[str, nn.Module]) -> None:
    """ Scales the following input quantization observer's min/max values by
    updating the values with the scaled min/max values calculated by the input
    equalization observer
    """
    input_eq_obs = modules[str(node.target)]
    assert isinstance(input_eq_obs, _InputEqualizationObserver)
    input_quant_obs_node = node.args[0]
    assert isinstance(input_quant_obs_node, Node)
    input_quant_obs = modules[str(input_quant_obs_node.target)]
    if not isinstance(input_quant_obs, ObserverBase):
        return
    min_input_scaled, max_input_scaled = input_eq_obs.calculate_scaled_minmax()
    if min_input_scaled is None and max_input_scaled is None:
        return
    input_quant_obs.min_val = min_input_scaled
    input_quant_obs.max_val = max_input_scaled