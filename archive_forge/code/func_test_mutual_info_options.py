import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_mutual_info_options(global_dtype, csr_container):
    X = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype)
    y = np.array([0, 1, 2, 2, 1], dtype=global_dtype)
    X_csr = csr_container(X)
    for mutual_info in (mutual_info_regression, mutual_info_classif):
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=False)
        with pytest.raises(ValueError):
            mutual_info(X, y, discrete_features='manual')
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=[True, False, True])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[True, False, True, False])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[1, 4])
        mi_1 = mutual_info(X, y, discrete_features='auto', random_state=0)
        mi_2 = mutual_info(X, y, discrete_features=False, random_state=0)
        mi_3 = mutual_info(X_csr, y, discrete_features='auto', random_state=0)
        mi_4 = mutual_info(X_csr, y, discrete_features=True, random_state=0)
        mi_5 = mutual_info(X, y, discrete_features=[True, False, True], random_state=0)
        mi_6 = mutual_info(X, y, discrete_features=[0, 2], random_state=0)
        assert_allclose(mi_1, mi_2)
        assert_allclose(mi_3, mi_4)
        assert_allclose(mi_5, mi_6)
        assert not np.allclose(mi_1, mi_3)