from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten.is_non_overlapping_and_dense])
def is_non_overlapping_and_dense(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError('MaskedTensors with sparse data do not have is_non_overlapping_and_dense')
    return func(data, *args[1:], **kwargs)