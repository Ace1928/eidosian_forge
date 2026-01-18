import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@exog_params.setter
def exog_params(self, value):
    if np.isscalar(value):
        value = [value] * self.k_exog_params
    self._params_split['exog_params'] = validate_basic(value, self.k_exog_params, title='exogenous coefficients')
    self._params = None