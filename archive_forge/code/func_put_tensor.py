from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
def put_tensor(self, tensor: Optional[Tensor], tensor_life: int) -> None:
    """Stores a tensor into this portal."""
    self.tensor_life = tensor_life
    if tensor_life > 0:
        self.tensor = tensor
    else:
        self.tensor = None