from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_linear(linear: nn.Linear) -> None:
    mask = _prune_linear_helper(linear)
    if getattr(linear, 'prune_bias', False):
        _prune_module_bias(linear, mask)