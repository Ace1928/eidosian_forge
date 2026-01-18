from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def proportions_ztost(count, nobs, low, upp, prop_var='sample'):
    """
    Equivalence test based on normal distribution

    Parameters
    ----------
    count : {int, array_like}
        the number of successes in nobs trials. If this is array_like, then
        the assumption is that this represents the number of successes for
        each independent sample
    nobs : int
        the number of trials or observations, with the same length as
        count.
    low, upp : float
        equivalence interval low < prop1 - prop2 < upp
    prop_var : str or float in (0, 1)
        prop_var determines which proportion is used for the calculation
        of the standard deviation of the proportion estimate
        The available options for string are 'sample' (default), 'null' and
        'limits'. If prop_var is a float, then it is used directly.

    Returns
    -------
    pvalue : float
        pvalue of the non-equivalence test
    t1, pv1 : tuple of floats
        test statistic and pvalue for lower threshold test
    t2, pv2 : tuple of floats
        test statistic and pvalue for upper threshold test

    Notes
    -----
    checked only for 1 sample case

    """
    if prop_var == 'limits':
        prop_var_low = low
        prop_var_upp = upp
    elif prop_var == 'sample':
        prop_var_low = prop_var_upp = False
    elif prop_var == 'null':
        prop_var_low = prop_var_upp = 0.5 * (low + upp)
    elif np.isreal(prop_var):
        prop_var_low = prop_var_upp = prop_var
    tt1 = proportions_ztest(count, nobs, alternative='larger', prop_var=prop_var_low, value=low)
    tt2 = proportions_ztest(count, nobs, alternative='smaller', prop_var=prop_var_upp, value=upp)
    return (np.maximum(tt1[1], tt2[1]), tt1, tt2)