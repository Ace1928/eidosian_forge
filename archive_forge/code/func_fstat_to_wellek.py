import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def fstat_to_wellek(f_stat, n_groups, nobs_mean):
    """Convert F statistic to wellek's effect size eps squared

    This computes the following effect size :

       es = f_stat * (n_groups - 1) / nobs_mean

    Parameters
    ----------
    f_stat : float or ndarray
        Test statistic of an F-test.
    n_groups : int
        Number of groups in oneway comparison
    nobs_mean : float or ndarray
        Average number of observations across groups.

    Returns
    -------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test

    """
    es = f_stat * (n_groups - 1) / nobs_mean
    return es