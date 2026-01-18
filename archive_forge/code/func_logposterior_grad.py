from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def logposterior_grad(self, params):
    """
        The gradient of the log posterior.
        """
    fep, vcp, vc = self._unpack(params)
    lp = 0
    if self.k_fep > 0:
        lp += np.dot(self.exog, fep)
    if self.k_vc > 0:
        lp += self.exog_vc.dot(vc)
    mu = self.family.link.inverse(lp)
    score_factor = (self.endog - mu) / self.family.link.deriv(mu)
    score_factor /= self.family.variance(mu)
    te = [None, None, None]
    if self.k_fep > 0:
        te[0] = np.dot(score_factor, self.exog)
    if self.k_vc > 0:
        te[2] = self.exog_vc.transpose().dot(score_factor)
    if self.k_vc > 0:
        vcp0 = vcp[self.ident]
        s = np.exp(vcp0)
        u = vc ** 2 / s ** 2 - 1
        te[1] = np.bincount(self.ident, weights=u)
        te[2] -= vc / s ** 2
        te[1] -= vcp / self.vcp_p ** 2
    if self.k_fep > 0:
        te[0] -= fep / self.fe_p ** 2
    te = [x for x in te if x is not None]
    return np.concatenate(te)