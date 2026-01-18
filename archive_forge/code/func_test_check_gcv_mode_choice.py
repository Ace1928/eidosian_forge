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
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
@pytest.mark.parametrize('mode, mode_n_greater_than_p, mode_p_greater_than_n', [(None, 'svd', 'eigen'), ('auto', 'svd', 'eigen'), ('eigen', 'eigen', 'eigen'), ('svd', 'svd', 'svd')])
def test_check_gcv_mode_choice(sparse_container, mode, mode_n_greater_than_p, mode_p_greater_than_n):
    X, _ = make_regression(n_samples=5, n_features=2)
    if sparse_container is not None:
        X = sparse_container(X)
    assert _check_gcv_mode(X, mode) == mode_n_greater_than_p
    assert _check_gcv_mode(X.T, mode) == mode_p_greater_than_n