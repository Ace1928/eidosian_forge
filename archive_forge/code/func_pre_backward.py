from contextlib import nullcontext
from typing import Any, ContextManager, Dict, Literal, Optional, Union
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from lightning_fabric.utilities.types import _PARAMETERS, Optimizable
def pre_backward(self, tensor: Tensor, module: Optional[Module]) -> Any:
    """Runs before precision plugin executes backward.

        Args:
            tensor: The tensor that will be used for backpropagation
            module: The module that was involved in producing the tensor and whose parameters need the gradients

        """