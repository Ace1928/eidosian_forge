from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
def orange(self, phony: Tensor) -> Optional[Tensor]:
    """Creates a :class:`PortalOrange` which retrieves the hidden tensor
        without losing ability of backpropagation.

        Give a phony forked from the main lane of an autograd graph::

                +-- PortalOrange --+
                |                  |
            -- Fork --------- f(a, b) --

        """
    self.check_tensor_life()
    if self.tensor is None:
        return self.use_tensor()
    return PortalOrange.apply(self, phony)