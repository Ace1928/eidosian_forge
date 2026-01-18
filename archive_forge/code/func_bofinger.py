import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
def bofinger(n, q):
    num = 9.0 / 2 * norm.pdf(2 * norm.ppf(q)) ** 4
    den = (2 * norm.ppf(q) ** 2 + 1) ** 2
    h = n ** (-1.0 / 5) * (num / den) ** (1.0 / 5)
    return h