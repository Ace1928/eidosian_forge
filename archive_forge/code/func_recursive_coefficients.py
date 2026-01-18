import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
@property
def recursive_coefficients(self):
    """
        Estimates of regression coefficients, recursively estimated

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
    out = None
    spec = self.specification
    start = offset = 0
    end = offset + spec.k_exog
    out = Bunch(filtered=self.filtered_state[start:end], filtered_cov=self.filtered_state_cov[start:end, start:end], smoothed=None, smoothed_cov=None, offset=offset)
    if self.smoothed_state is not None:
        out.smoothed = self.smoothed_state[start:end]
    if self.smoothed_state_cov is not None:
        out.smoothed_cov = self.smoothed_state_cov[start:end, start:end]
    return out