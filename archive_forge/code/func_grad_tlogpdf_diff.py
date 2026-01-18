from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import psi
def grad_tlogpdf_diff(diff, df):
    return -diff * (1.0 + df) / (diff ** 2 + df)