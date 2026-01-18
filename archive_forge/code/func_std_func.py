import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
def std_func(x):
    """standard deviation function for example"""
    return 0.1 * np.exp(2.5 + 0.75 * np.abs(x))