from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def start_weights(self, inv=True):
    """Starting weights"""
    zz = np.dot(self.instrument.T, self.instrument)
    nobs = self.instrument.shape[0]
    if inv:
        return zz / nobs
    else:
        return np.linalg.pinv(zz / nobs)