from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
def objective_func(x):
    constant_pred = np.full(n_samples, fill_value=x)
    return mean_pinball_loss(data, constant_pred, alpha=target_quantile)