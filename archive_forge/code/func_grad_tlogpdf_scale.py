from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import psi
def grad_tlogpdf_scale(x, df, loc, scale):
    diff = x - loc
    return -(df * (scale ** 2 - diff ** 2)) / (scale * (df * scale ** 2 + diff ** 2))