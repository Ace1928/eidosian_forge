import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@ma_poly.setter
def ma_poly(self, value):
    if isinstance(value, Polynomial):
        value = value.coef
    value = validate_basic(value, self.spec.max_ma_order + 1, title='MA polynomial')
    if value[0] != 1:
        raise ValueError('MA polynomial constant must be equal to 1.')
    ma_params = []
    for i in range(1, self.spec.max_ma_order + 1):
        if i in self.spec.ma_lags:
            ma_params.append(value[i])
        elif value[i] != 0:
            raise ValueError('MA polynomial includes non-zero values for lags that are excluded in the specification.')
    self.ma_params = ma_params