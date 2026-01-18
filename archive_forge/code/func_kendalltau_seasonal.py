import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def kendalltau_seasonal(x):
    """
    Computes a multivariate Kendall's rank correlation tau, for seasonal data.

    Parameters
    ----------
    x : 2-D ndarray
        Array of seasonal data, with seasons in columns.

    """
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    n, m = x.shape
    n_p = x.count(0)
    S_szn = sum((msign(x[i:] - x[i]).sum(0) for i in range(n)))
    S_tot = S_szn.sum()
    n_tot = x.count()
    ties = count_tied_groups(x.compressed())
    corr_ties = sum((v * k * (k - 1) for k, v in ties.items()))
    denom_tot = ma.sqrt(1.0 * n_tot * (n_tot - 1) * (n_tot * (n_tot - 1) - corr_ties)) / 2.0
    R = rankdata(x, axis=0, use_missing=True)
    K = ma.empty((m, m), dtype=int)
    covmat = ma.empty((m, m), dtype=float)
    denom_szn = ma.empty(m, dtype=float)
    for j in range(m):
        ties_j = count_tied_groups(x[:, j].compressed())
        corr_j = sum((v * k * (k - 1) for k, v in ties_j.items()))
        cmb = n_p[j] * (n_p[j] - 1)
        for k in range(j, m, 1):
            K[j, k] = sum((msign((x[i:, j] - x[i, j]) * (x[i:, k] - x[i, k])).sum() for i in range(n)))
            covmat[j, k] = (K[j, k] + 4 * (R[:, j] * R[:, k]).sum() - n * (n_p[j] + 1) * (n_p[k] + 1)) / 3.0
            K[k, j] = K[j, k]
            covmat[k, j] = covmat[j, k]
        denom_szn[j] = ma.sqrt(cmb * (cmb - corr_j)) / 2.0
    var_szn = covmat.diagonal()
    z_szn = msign(S_szn) * (abs(S_szn) - 1) / ma.sqrt(var_szn)
    z_tot_ind = msign(S_tot) * (abs(S_tot) - 1) / ma.sqrt(var_szn.sum())
    z_tot_dep = msign(S_tot) * (abs(S_tot) - 1) / ma.sqrt(covmat.sum())
    prob_szn = special.erfc(abs(z_szn) / np.sqrt(2))
    prob_tot_ind = special.erfc(abs(z_tot_ind) / np.sqrt(2))
    prob_tot_dep = special.erfc(abs(z_tot_dep) / np.sqrt(2))
    chi2_tot = (z_szn * z_szn).sum()
    chi2_trd = m * z_szn.mean() ** 2
    output = {'seasonal tau': S_szn / denom_szn, 'global tau': S_tot / denom_tot, 'global tau (alt)': S_tot / denom_szn.sum(), 'seasonal p-value': prob_szn, 'global p-value (indep)': prob_tot_ind, 'global p-value (dep)': prob_tot_dep, 'chi2 total': chi2_tot, 'chi2 trend': chi2_trd}
    return output