from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def msear_err(arma, ar_des):
    ar, ma = (np.r_[1, arma[:p - 1]], np.r_[1, arma[p - 1:]])
    ar_approx = arma_impulse_response(ma, ar, n)
    return ar_des - ar_approx