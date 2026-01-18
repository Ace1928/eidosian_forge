from typing import List, Optional, Sequence, Tuple, Union
import torch
from .common import _get_storage_base
def stack_or_none(tensors: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Does exactly the same as :attr:`torch.stack` if the tensors can be concatenated
    without any memory operation. Otherwise returns None.
    """
    return _StackOrNone.apply(dim, *tensors)