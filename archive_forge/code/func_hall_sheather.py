import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
def hall_sheather(n, q, alpha=0.05):
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z) ** 2.0
    den = 2.0 * z ** 2.0 + 1.0
    h = n ** (-1.0 / 3) * norm.ppf(1.0 - alpha / 2.0) ** (2.0 / 3) * (num / den) ** (1.0 / 3)
    return h