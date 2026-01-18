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
def logposterior(self, params):
    """
        The overall log-density: log p(y, fe, vc, vcp).

        This differs by an additive constant from the log posterior
        log p(fe, vc, vcp | y).
        """
    fep, vcp, vc = self._unpack(params)
    lp = 0
    if self.k_fep > 0:
        lp += np.dot(self.exog, fep)
    if self.k_vc > 0:
        lp += self.exog_vc.dot(vc)
    mu = self.family.link.inverse(lp)
    ll = self.family.loglike(self.endog, mu)
    if self.k_vc > 0:
        vcp0 = vcp[self.ident]
        s = np.exp(vcp0)
        ll -= 0.5 * np.sum(vc ** 2 / s ** 2) + np.sum(vcp0)
        ll -= 0.5 * np.sum(vcp ** 2 / self.vcp_p ** 2)
    if self.k_fep > 0:
        ll -= 0.5 * np.sum(fep ** 2 / self.fe_p ** 2)
    return ll