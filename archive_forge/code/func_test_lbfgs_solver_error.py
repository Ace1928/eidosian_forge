import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_lbfgs_solver_error():
    """Test that LBFGS solver raises ConvergenceWarning."""
    X = np.array([[1, -1], [1, 1]])
    y = np.array([-10000000000.0, 10000000000.0])
    model = Ridge(alpha=0.01, solver='lbfgs', fit_intercept=False, tol=1e-12, positive=True, max_iter=1)
    with pytest.warns(ConvergenceWarning, match='lbfgs solver did not converge'):
        model.fit(X, y)