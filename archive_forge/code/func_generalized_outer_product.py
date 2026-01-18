from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.extend import primitive, defvjp
def generalized_outer_product(x):
    if np.ndim(x) == 1:
        return np.outer(x, x)
    return np.matmul(x, np.swapaxes(x, -1, -2))