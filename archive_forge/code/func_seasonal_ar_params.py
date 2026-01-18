import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@seasonal_ar_params.setter
def seasonal_ar_params(self, value):
    if np.isscalar(value):
        value = [value] * self.k_seasonal_ar_params
    self._params_split['seasonal_ar_params'] = validate_basic(value, self.k_seasonal_ar_params, title='seasonal AR coefficients')
    self._params = None