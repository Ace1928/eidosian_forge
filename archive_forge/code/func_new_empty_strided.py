from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten.new_empty_strided])
def new_empty_strided(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f'__torch_dispatch__, {func}', len_args=3)
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    if tuple(args[1]) != tuple(data.size()):
        raise ValueError(f'__torch_dispatch__, {func}: args[1] expected to be the same as data.size()')
    if tuple(args[2]) != tuple(data.stride()):
        raise ValueError(f'__torch_dispatch__, {func}: args[2] expected to be the same as data.stride()')
    return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)