import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal
def model_args(x):
    return (x[:, 0], x[:, 1])