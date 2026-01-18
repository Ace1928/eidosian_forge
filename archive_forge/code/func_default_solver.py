import numpy as np
import pytest
from pytest import approx
from scipy.optimize import minimize
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.utils._testing import assert_allclose, skip_if_32bit
from sklearn.utils.fixes import (
@pytest.fixture
def default_solver():
    return 'highs' if sp_version >= parse_version('1.6.0') else 'interior-point'