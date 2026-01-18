import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_compute_mi_cd_unique_label(global_dtype):
    n_samples = 100
    x = np.random.uniform(size=n_samples) > 0.5
    y = np.empty(n_samples, global_dtype)
    mask = x == 0
    y[mask] = np.random.uniform(-1, 1, size=np.sum(mask))
    y[~mask] = np.random.uniform(0, 2, size=np.sum(~mask))
    mi_1 = _compute_mi(x, y, x_discrete=True, y_discrete=False)
    x = np.hstack((x, 2))
    y = np.hstack((y, 10))
    mi_2 = _compute_mi(x, y, x_discrete=True, y_discrete=False)
    assert_allclose(mi_1, mi_2)