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
def power_binom_tost(low, upp, nobs, p_alt=None, alpha=0.05):
    if p_alt is None:
        p_alt = 0.5 * (low + upp)
    x_low, x_upp = binom_tost_reject_interval(low, upp, nobs, alpha=alpha)
    power = stats.binom.cdf(x_upp, nobs, p_alt) - stats.binom.cdf(x_low - 1, nobs, p_alt)
    return power