import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
@cache_readonly
def llf_recursive_obs(self):
    """
        (float) Loglikelihood at observation, computed from recursive residuals
        """
    from scipy.stats import norm
    return np.log(norm.pdf(self.resid_recursive, loc=0, scale=self.scale ** 0.5))