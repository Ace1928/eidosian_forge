from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import psi
def grad_tlogpdf_df(x, df, loc, scale):
    y = (x - loc) / scale
    return 0.5 * (y ** 2 * (df + 1) / (df * (y ** 2 + df)) - np.log(y ** 2 / df + 1) - 1.0 / df - psi(df / 2.0) + psi((df + 1) / 2.0))