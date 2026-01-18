import inspect
from typing import Callable, List, Optional, Set, Tuple, Union
import torch
from packaging import version
from safetensors.torch import storage_ptr, storage_size
from torch import nn
from .utils import is_torch_tpu_available, logging
def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    A function that calls the internal `_softmax_backward_data` PyTorch method and that adjusts the arguments according
    to the torch version detected.
    """
    from torch import _softmax_backward_data
    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)