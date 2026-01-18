from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def set_param_names(self, param_names, k_params=None):
    """set the parameter names in the model

        Parameters
        ----------
        param_names : list[str]
            param_names should have the same length as the number of params
        k_params : None or int
            If k_params is None, then the k_params attribute is used, unless
            it is None.
            If k_params is not None, then it will also set the k_params
            attribute.
        """
    if k_params is not None:
        self.k_params = k_params
    else:
        k_params = self.k_params
    if k_params == len(param_names):
        self.data.xnames = param_names
    else:
        raise ValueError('param_names has the wrong length')