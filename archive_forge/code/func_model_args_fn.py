import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal
def model_args_fn(x):
    if type(x) is np.ndarray:
        return (x[:, 0], x[:, 1:])
    else:
        return (x.iloc[:, 0].values, x.iloc[:, 1:].values)