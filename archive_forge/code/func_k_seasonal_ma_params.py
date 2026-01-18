import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def k_seasonal_ma_params(self):
    """(int) Number of seasonal moving average parameters."""
    return len(self.seasonal_ma_lags)