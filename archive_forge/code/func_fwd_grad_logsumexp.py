from __future__ import absolute_import
import scipy.special
import autograd.numpy as np
from autograd.extend import primitive, defvjp, defjvp
from autograd.numpy.numpy_vjps import unbroadcast_f, repeat_to_match_shape
def fwd_grad_logsumexp(g, ans, x, axis=None, b=1.0, keepdims=False):
    if not keepdims:
        if isinstance(axis, int):
            ans = np.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = np.expand_dims(ans, ax)
    return np.sum(g * b * np.exp(x - ans), axis=axis, keepdims=keepdims)