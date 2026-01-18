from __future__ import absolute_import
import scipy.special
import autograd.numpy as np
from autograd.extend import primitive, defvjp, defjvp
from autograd.numpy.numpy_vjps import unbroadcast_f, repeat_to_match_shape
def make_grad_logsumexp(ans, x, axis=None, b=1.0, keepdims=False):
    shape, dtype = (np.shape(x), np.result_type(x))

    def vjp(g):
        g_repeated, _ = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        ans_repeated, _ = repeat_to_match_shape(ans, shape, dtype, axis, keepdims)
        return g_repeated * b * np.exp(x - ans_repeated)
    return vjp