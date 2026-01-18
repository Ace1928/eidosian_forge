from __future__ import absolute_import
import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
def grad_poisson_logpmf(k, mu):
    return np.where(k % 1 == 0, k / mu - 1, 0)