import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional, cast
import torch
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
def clip_grad_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float=2.0, error_if_nonfinite: bool=False, foreach: Optional[bool]=None) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    warnings.warn('torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.', stacklevel=2)
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)