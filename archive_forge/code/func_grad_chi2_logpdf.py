from __future__ import absolute_import, division
import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import gamma
def grad_chi2_logpdf(x, df):
    return np.where(df % 1 == 0, (df - x - 2) / (2 * x), 0)