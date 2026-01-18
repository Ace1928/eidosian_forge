from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
def put_grad(self, grad: Tensor) -> None:
    """Stores a gradient into this portal."""
    self.grad = grad