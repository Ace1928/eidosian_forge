from contextlib import nullcontext
from typing import Any, ContextManager, Dict, Literal, Optional, Union
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from lightning_fabric.utilities.types import _PARAMETERS, Optimizable
def main_params(self, optimizer: Optimizer) -> _PARAMETERS:
    """The main params of the model.

        Returns the plain model params here. Maybe different in other precision plugins.

        """
    for group in optimizer.param_groups:
        yield from group['params']