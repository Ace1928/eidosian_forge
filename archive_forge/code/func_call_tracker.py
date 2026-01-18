from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
@torch.jit.unused
def call_tracker(self):
    """Interface for tracking iteration process in Python mode.

        Tracking the iteration process is disabled in TorchScript
        mode. In fact, one should specify tracker=None when JIT
        compiling functions using lobpcg.
        """
    pass