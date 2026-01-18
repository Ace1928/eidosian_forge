import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_compute_mi_cd(global_dtype):
    n_samples = 1000
    rng = check_random_state(0)
    for p in [0.3, 0.5, 0.7]:
        x = rng.uniform(size=n_samples) > p
        y = np.empty(n_samples, global_dtype)
        mask = x == 0
        y[mask] = rng.uniform(-1, 1, size=np.sum(mask))
        y[~mask] = rng.uniform(0, 2, size=np.sum(~mask))
        I_theory = -0.5 * ((1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)) - np.log(2)
        for n_neighbors in [3, 5, 7]:
            I_computed = _compute_mi(x, y, x_discrete=True, y_discrete=False, n_neighbors=n_neighbors)
            assert_allclose(I_computed, I_theory, rtol=0.1)