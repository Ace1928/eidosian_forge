from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def momcond_mean(self, params):
    """
        mean of moment conditions,

        """
    momcond = self.momcond(params)
    self.nobs_moms, self.k_moms = momcond.shape
    return momcond.mean(0)