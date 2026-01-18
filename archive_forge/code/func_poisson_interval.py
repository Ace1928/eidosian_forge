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
def poisson_interval(interval, p):
    """
            Compute P(b <= Z <= a) where Z ~ Poisson(p) and
            `interval = (b, a)`.
            """
    b, a = interval
    prob = stats.poisson.cdf(a, p) - stats.poisson.cdf(b - 1, p)
    return prob