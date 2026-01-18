from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def score_cu(self, params, epsilon=None, centered=True):
    """Score cu"""
    deriv = approx_fprime(params, self.gmmobjective_cu, args=(), centered=centered, epsilon=epsilon)
    return deriv