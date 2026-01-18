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
@cache_readonly
def random_effects_cov(self):
    """
        Returns the conditional covariance matrix of the random
        effects for each group given the data.

        Returns
        -------
        random_effects_cov : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """
    try:
        cov_re_inv = np.linalg.inv(self.cov_re)
    except np.linalg.LinAlgError:
        cov_re_inv = None
    vcomp = self.vcomp
    ranef_dict = {}
    for group_ix in range(self.model.n_groups):
        ex_r = self.model._aex_r[group_ix]
        ex2_r = self.model._aex_r2[group_ix]
        label = self.model.group_labels[group_ix]
        vc_var = self.model._expand_vcomp(vcomp, group_ix)
        solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
        n = ex_r.shape[0]
        m = self.cov_re.shape[0]
        mat1 = np.empty((n, m + len(vc_var)))
        mat1[:, 0:m] = np.dot(ex_r[:, 0:m], self.cov_re)
        mat1[:, m:] = np.dot(ex_r[:, m:], np.diag(vc_var))
        mat2 = solver(mat1)
        mat2 = np.dot(mat1.T, mat2)
        v = -mat2
        v[0:m, 0:m] += self.cov_re
        ix = np.arange(m, v.shape[0])
        v[ix, ix] += vc_var
        na = self._expand_re_names(group_ix)
        v = pd.DataFrame(v, index=na, columns=na)
        ranef_dict[label] = v
    return ranef_dict