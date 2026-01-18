from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
@cache_readonly
def fvalue(self):
    const_idx = self.model.data.const_idx
    if const_idx is None:
        return np.nan
    else:
        k_vars = len(self.params)
        restriction = np.eye(k_vars)
        idx_noconstant = lrange(k_vars)
        del idx_noconstant[const_idx]
        fval = self.f_test(restriction[idx_noconstant]).fvalue
        return fval