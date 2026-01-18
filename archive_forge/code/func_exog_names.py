import numpy as np
import pandas as pd
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima.tools import standardize_lag_order, validate_basic
@property
def exog_names(self):
    """(list of str) Names associated with exogenous parameters."""
    exog_names = self._model.exog_names
    return [] if exog_names is None else exog_names