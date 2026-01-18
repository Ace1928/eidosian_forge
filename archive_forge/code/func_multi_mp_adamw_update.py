import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
def multi_mp_adamw_update(weights, grads, mean, var, weights32, rescale_grad, lrs, wds, etas, out=None, name=None, size=0, **kwargs):
    if not size:
        size = len(weights)
    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weights[0].context)
    temp_list = _flatten_list(zip(weights, grads, mean, var, weights32)) + [rescale_grad]
    return ndarray._internal._multi_mp_adamw_update(*temp_list, out=out, num_weights=size, lrs=lrs, wds=wds, etas=etas, name=name, **kwargs)