import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
def mp_adamw_update(weight, grad, mean, var, weight32, rescale_grad, lr, eta, beta1=0.9, beta2=0.999, epsilon=1e-08, wd=0, clip_gradient=-1, out=None, name=None, **kwargs):
    rescale_grad = _get_rescale_grad(rescale_grad, ctx=weight.context)
    return ndarray._internal._mp_adamw_update(weight=weight, grad=grad, mean=mean, var=var, weight32=weight32, rescale_grad=rescale_grad, lr=lr, eta=eta, beta1=beta1, beta2=beta2, epsilon=epsilon, wd=wd, clip_gradient=clip_gradient, out=out, name=name, **kwargs)