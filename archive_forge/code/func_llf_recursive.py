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
def llf_recursive(self):
    """
        (float) Loglikelihood defined by recursive residuals, equivalent to OLS
        """
    return np.sum(self.llf_recursive_obs)