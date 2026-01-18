import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@seasonal_ar_poly.setter
def seasonal_ar_poly(self, value):
    s = self.spec.seasonal_periods
    if isinstance(value, Polynomial):
        value = value.coef
    value = validate_basic(value, 1 + s * self.spec.max_seasonal_ar_order, title='seasonal AR polynomial')
    if value[0] != 1:
        raise ValueError('Polynomial constant must be equal to 1.')
    seasonal_ar_params = []
    for i in range(1, self.spec.max_seasonal_ar_order + 1):
        if i in self.spec.seasonal_ar_lags:
            seasonal_ar_params.append(-value[s * i])
        elif value[s * i] != 0:
            raise ValueError('AR polynomial includes non-zero values for lags that are excluded in the specification.')
    self.seasonal_ar_params = seasonal_ar_params