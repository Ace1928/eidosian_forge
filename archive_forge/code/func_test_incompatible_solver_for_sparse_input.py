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
@pytest.mark.skipif(parse_version(sp_version.base_version) >= parse_version('1.11'), reason='interior-point solver is not available in SciPy 1.11')
@pytest.mark.parametrize('solver', ['interior-point', 'revised simplex'])
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_incompatible_solver_for_sparse_input(X_y_data, solver, csc_container):
    X, y = X_y_data
    X_sparse = csc_container(X)
    err_msg = f"Solver {solver} does not support sparse X. Use solver 'highs' for example."
    with pytest.raises(ValueError, match=err_msg):
        QuantileRegressor(solver=solver).fit(X_sparse, y)