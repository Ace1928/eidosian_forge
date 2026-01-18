import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('correlated', [True, False])
def test_mutual_information_symmetry_classif_regression(correlated, global_random_seed):
    """Check that `mutual_info_classif` and `mutual_info_regression` are
    symmetric by switching the target `y` as `feature` in `X` and vice
    versa.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23720
    """
    rng = np.random.RandomState(global_random_seed)
    n = 100
    d = rng.randint(10, size=n)
    if correlated:
        c = d.astype(np.float64)
    else:
        c = rng.normal(0, 1, size=n)
    mi_classif = mutual_info_classif(c[:, None], d, discrete_features=[False], random_state=global_random_seed)
    mi_regression = mutual_info_regression(d[:, None], c, discrete_features=[True], random_state=global_random_seed)
    assert mi_classif == pytest.approx(mi_regression)