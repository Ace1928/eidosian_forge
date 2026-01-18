import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def detrend_linear(y):
    """Return y minus best fit line; 'linear' detrending """
    x = np.arange(len(y), dtype=np.float_)
    C = np.cov(x, y, bias=1)
    b = C[0, 1] / C[0, 0]
    a = y.mean() - b * x.mean()
    return y - (b * x + a)