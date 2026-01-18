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
def binom_test_reject_interval(value, nobs, alpha=0.05, alternative='two-sided'):
    """
    Rejection region for binomial test for one sample proportion

    The interval includes the end points of the rejection region.

    Parameters
    ----------
    value : float
        proportion under the Null hypothesis
    nobs : int
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : int
        lower and upper bound of rejection region
    """
    if alternative in ['2s', 'two-sided']:
        alternative = '2s'
        alpha = alpha / 2
    if alternative in ['2s', 'smaller']:
        x_low = stats.binom.ppf(alpha, nobs, value) - 1
    else:
        x_low = 0
    if alternative in ['2s', 'larger']:
        x_upp = stats.binom.isf(alpha, nobs, value) + 1
    else:
        x_upp = nobs
    return (int(x_low), int(x_upp))