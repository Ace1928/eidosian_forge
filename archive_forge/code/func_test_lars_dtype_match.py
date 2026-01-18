import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
@pytest.mark.parametrize('LARS, has_coef_path, args', ((Lars, True, {}), (LassoLars, True, {}), (LassoLarsIC, False, {}), (LarsCV, True, {}), (LassoLarsCV, True, {'max_iter': 5})))
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_lars_dtype_match(LARS, has_coef_path, args, dtype):
    rng = np.random.RandomState(0)
    X = rng.rand(20, 6).astype(dtype)
    y = rng.rand(20).astype(dtype)
    model = LARS(**args)
    model.fit(X, y)
    assert model.coef_.dtype == dtype
    if has_coef_path:
        assert model.coef_path_.dtype == dtype
    assert model.intercept_.dtype == dtype