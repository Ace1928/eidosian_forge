from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten.copy_])
def copy_(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f'__torch_dispatch__, {func}', len_args=2)
    if not _masks_match(_maybe_get_mask(args[0]), _maybe_get_mask(args[1])):
        raise ValueError('args[0] mask and args[1] mask must match but do not')
    func(_get_data(args[0]), _get_data(args[1]))
    return args[0]