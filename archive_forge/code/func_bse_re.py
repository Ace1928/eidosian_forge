import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
@cache_readonly
def bse_re(self):
    """
        Returns the standard errors of the variance parameters.

        The first `k_re x (k_re + 1)` elements of the returned array
        are the standard errors of the lower triangle of `cov_re`.
        The remaining elements are the standard errors of the variance
        components.

        Note that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        or p-values if used in the usual way.
        """
    p = self.model.exog.shape[1]
    return np.sqrt(self.scale * np.diag(self.cov_params())[p:])