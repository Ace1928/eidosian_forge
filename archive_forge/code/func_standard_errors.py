import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def standard_errors(self):
    """
        Returns the standard errors of the parameter estimates.
        """
    return np.sqrt(np.diag(self.cov_params()))