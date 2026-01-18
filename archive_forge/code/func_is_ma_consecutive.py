import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def is_ma_consecutive(self):
    """
        (bool) Is moving average lag polynomial consecutive.

        I.e. does it include all lags up to and including the maximum lag.
        """
    return self.max_seasonal_ma_order == 0 and (not isinstance(self.ma_order, list))