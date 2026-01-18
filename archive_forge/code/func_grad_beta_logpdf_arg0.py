from __future__ import absolute_import
import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import beta, psi
def grad_beta_logpdf_arg0(x, a, b):
    return (1 + a * (x - 1) + x * (b - 2)) / (x * (x - 1))