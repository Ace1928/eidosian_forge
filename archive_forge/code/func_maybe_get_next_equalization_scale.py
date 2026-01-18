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
def maybe_get_next_equalization_scale(node: Node, modules: Dict[str, nn.Module]) -> Optional[torch.Tensor]:
    """ If the next next node is an InputEqualizationObserver then we want to
    return its equalization scale, else we return 1

    This is used in the case where there are two connecting linear layers:
        linear1 -> LinearOutObs -> InputEqObs -> linear2
    In this case, the node given is linear1 and we want to locate the InputEqObs.
    """
    next_inp_eq_obs = maybe_get_next_input_eq_obs(node, modules)
    if next_inp_eq_obs:
        if next_inp_eq_obs.equalization_scale.nelement() == 1 and next_inp_eq_obs.equalization_scale == torch.tensor(1):
            return None
        return next_inp_eq_obs.equalization_scale
    return None