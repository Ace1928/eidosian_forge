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
def reshape_scale(scale: torch.Tensor, axis: int, input: torch.Tensor) -> torch.Tensor:
    """Reshapes the scale so that we can multiply it to the input by the given axis.
    """
    new_shape = [1] * input.ndim
    new_shape[axis] = input.size(axis)
    return scale.view(new_shape)