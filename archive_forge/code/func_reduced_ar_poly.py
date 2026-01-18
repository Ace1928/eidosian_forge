import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.statespace.tools import is_invertible
from statsmodels.tsa.arima.tools import validate_basic
@property
def reduced_ar_poly(self):
    """(Polynomial) Reduced form autoregressive lag polynomial."""
    return self.ar_poly * self.seasonal_ar_poly