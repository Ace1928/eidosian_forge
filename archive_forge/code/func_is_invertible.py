import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@property
def is_invertible(self):
    """(bool) Is the reduced moving average lag poylnomial invertible."""
    validate_basic(self.ma_params, self.k_ma_params, title='MA coefficients')
    validate_basic(self.seasonal_ma_params, self.k_seasonal_ma_params, title='seasonal MA coefficients')
    ma_stationary = True
    seasonal_ma_stationary = True
    if self.k_ma_params > 0:
        ma_stationary = is_invertible(self.ma_poly.coef)
    if self.k_seasonal_ma_params > 0:
        seasonal_ma_stationary = is_invertible(self.seasonal_ma_poly.coef)
    return ma_stationary and seasonal_ma_stationary