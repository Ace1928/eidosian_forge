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
def binom_tost_reject_interval(low, upp, nobs, alpha=0.05):
    """
    Rejection region for binomial TOST

    The interval includes the end points,
    `reject` if and only if `r_low <= x <= r_upp`.

    The interval might be empty with `r_upp < r_low`.

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        the number of trials or observations.

    Returns
    -------
    x_low, x_upp : float
        lower and upper bound of rejection region

    """
    x_low = stats.binom.isf(alpha, nobs, low) + 1
    x_upp = stats.binom.ppf(alpha, nobs, upp) - 1
    return (x_low, x_upp)