import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def ma_names(self):
    """(list of str) Names of (non-seasonal) moving average parameters."""
    return ['ma.L%d' % i for i in self.ma_lags]