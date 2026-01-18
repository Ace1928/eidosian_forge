import copy
from typing import Optional, Any, Union, Callable
import torch
import warnings
from torch import Tensor
from .. import functional as F
from .module import Module
from .activation import MultiheadAttention
from .container import ModuleList
from ..init import xavier_uniform_
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm
@staticmethod
def generate_square_subsequent_mask(sz: int, device: torch.device=torch.device(torch._C._get_default_device()), dtype: torch.dtype=torch.get_default_dtype()) -> Tensor:
    """Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
    return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)