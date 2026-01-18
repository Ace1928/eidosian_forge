from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_linear_linear(linear1: nn.Linear, linear2: nn.Linear) -> None:
    prune_linear_activation_linear(linear1, None, linear2)