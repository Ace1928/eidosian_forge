from typing import Optional, Sequence
import torch
from xformers.ops._triton import (
from .common import BaseOperator, register_operator
def scaled_index_add(input: torch.Tensor, index: torch.Tensor, source: torch.Tensor, scaling: Optional[torch.Tensor]=None, alpha: float=1.0) -> torch.Tensor:
    """
    In-place scaling+index_add

    Indices in ``index`` are assumed to be unique

    The max index in ``index`` is assumed to be less than the size of dim0 of ``input``.

    :Note:

        The FW pass is done in-place (``input`` is modified)

    :Equivalent pytorch code:

    .. code-block:: python

        return torch.index_add(input, dim=0, source=scaling * src, index=indices, alpha=alpha)
    """
    return _ScaledIndexAdd.apply(input, index, source, scaling, alpha)