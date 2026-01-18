from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def distance_st_range(mean_all, nobs_all, var_all, df=None, triu=False):
    """pairwise distance matrix, outsourced from tukeyhsd



    CHANGED: meandiffs are with sign, studentized range uses abs

    q_crit added for testing

    TODO: error in variance calculation when nobs_all is scalar, missing 1/n

    """
    mean_all = np.asarray(mean_all)
    n_means = len(mean_all)
    if df is None:
        df = nobs_all - 1
    if np.size(df) == 1:
        df_total = n_means * df
    else:
        df_total = np.sum(df)
    if np.size(nobs_all) == 1 and np.size(var_all) == 1:
        var_pairs = 1.0 * var_all / nobs_all * np.ones((n_means, n_means))
    elif np.size(var_all) == 1:
        var_pairs = var_all * varcorrection_pairs_unbalanced(nobs_all, srange=True)
    elif np.size(var_all) > 1:
        var_pairs, df_sum = varcorrection_pairs_unequal(nobs_all, var_all, df)
        var_pairs /= 2.0
    else:
        raise ValueError('not supposed to be here')
    meandiffs = mean_all - mean_all[:, None]
    std_pairs = np.sqrt(var_pairs)
    idx1, idx2 = np.triu_indices(n_means, 1)
    if triu:
        meandiffs = meandiffs_[idx1, idx2]
        std_pairs = std_pairs_[idx1, idx2]
    st_range = np.abs(meandiffs) / std_pairs
    return (st_range, meandiffs, std_pairs, (idx1, idx2))