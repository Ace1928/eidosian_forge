import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def demean(x, axis=0):
    """Return x minus its mean along the specified axis"""
    x = np.asarray(x)
    if axis:
        ind = [slice(None)] * axis
        ind.append(np.newaxis)
        return x - x.mean(axis)[ind]
    return x - x.mean(axis)