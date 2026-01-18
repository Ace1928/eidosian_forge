from warnings import warn
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat
from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
@property
def initial_design(self):
    """Initial design matrix"""
    design = np.r_[[1] * self._k_diff, ([0] * (self.seasonal_periods - 1) + [1]) * self._k_seasonal_diff, [1] * self.state_error, [0] * (self._k_order - 1)]
    if len(design) == 0:
        design = np.r_[0]
    if self.state_regression:
        if self._k_order > 0:
            design = np.c_[np.reshape(np.repeat(design, self.nobs), (design.shape[0], self.nobs)).T, self.exog].T[None, :, :]
        else:
            design = self.exog.T[None, :, :]
    return design