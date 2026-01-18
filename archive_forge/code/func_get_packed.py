import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def get_packed(self, use_sqrt, has_fe=False):
    """
        Return the model parameters packed into a single vector.

        Parameters
        ----------
        use_sqrt : bool
            If True, the Cholesky square root of `cov_re` is
            included in the packed result.  Otherwise the
            lower triangle of `cov_re` is included.
        has_fe : bool
            If True, the fixed effects parameters are included
            in the packed result, otherwise they are omitted.
        """
    if self.k_re > 0:
        if use_sqrt:
            try:
                L = np.linalg.cholesky(self.cov_re)
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.diag(self.cov_re)))
            cpa = L[self._ix]
        else:
            cpa = self.cov_re[self._ix]
    else:
        cpa = np.zeros(0)
    if use_sqrt:
        vcomp = np.sqrt(self.vcomp)
    else:
        vcomp = self.vcomp
    if has_fe:
        pa = np.concatenate((self.fe_params, cpa, vcomp))
    else:
        pa = np.concatenate((cpa, vcomp))
    return pa