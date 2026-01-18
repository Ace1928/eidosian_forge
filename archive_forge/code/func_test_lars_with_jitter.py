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
@pytest.mark.parametrize('est', (LassoLars(alpha=0.001), Lars()))
def test_lars_with_jitter(est):
    X = np.array([[0.0, 0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0]])
    y = [-2.5, -2.5]
    expected_coef = [0, 2.5, 0, 2.5, 0]
    est.set_params(fit_intercept=False)
    est_jitter = clone(est).set_params(jitter=1e-07, random_state=0)
    est.fit(X, y)
    est_jitter.fit(X, y)
    assert np.mean((est.coef_ - est_jitter.coef_) ** 2) > 0.1
    np.testing.assert_allclose(est_jitter.coef_, expected_coef, rtol=0.001)