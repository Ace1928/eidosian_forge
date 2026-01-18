import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@ar_poly.setter
def ar_poly(self, value):
    if isinstance(value, Polynomial):
        value = value.coef
    value = validate_basic(value, self.spec.max_ar_order + 1, title='AR polynomial')
    if value[0] != 1:
        raise ValueError('AR polynomial constant must be equal to 1.')
    ar_params = []
    for i in range(1, self.spec.max_ar_order + 1):
        if i in self.spec.ar_lags:
            ar_params.append(-value[i])
        elif value[i] != 0:
            raise ValueError('AR polynomial includes non-zero values for lags that are excluded in the specification.')
    self.ar_params = ar_params