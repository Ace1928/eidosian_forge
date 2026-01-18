import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening
def exog_iterator():
    k_vars = 100
    n_batches = 6
    for i in range(n_batches):
        x = (0.05 * common + np.random.rand(nobs, k_vars) + 1.0 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
        x *= 1.2
        if i < k_nonzero - 1:
            x[:, 10] = x_nonzero[:, i + 1]
        x = (x - x.mean(0)) / x.std(0)
        yield x