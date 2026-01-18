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
def proportion_effectsize(prop1, prop2, method='normal'):
    """
    Effect size for a test comparing two proportions

    for use in power function

    Parameters
    ----------
    prop1, prop2 : float or array_like
        The proportion value(s).

    Returns
    -------
    es : float or ndarray
        effect size for (transformed) prop1 - prop2

    Notes
    -----
    only method='normal' is implemented to match pwr.p2.test
    see http://www.statmethods.net/stats/power.html

    Effect size for `normal` is defined as ::

        2 * (arcsin(sqrt(prop1)) - arcsin(sqrt(prop2)))

    I think other conversions to normality can be used, but I need to check.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> sm.stats.proportion_effectsize(0.5, 0.4)
    0.20135792079033088
    >>> sm.stats.proportion_effectsize([0.3, 0.4, 0.5], 0.4)
    array([-0.21015893,  0.        ,  0.20135792])

    """
    if method != 'normal':
        raise ValueError('only "normal" is implemented')
    es = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    return es