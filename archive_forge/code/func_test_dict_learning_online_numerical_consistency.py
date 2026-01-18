import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
@pytest.mark.parametrize('method', ('lars', 'cd'))
def test_dict_learning_online_numerical_consistency(method):
    rtol = 0.0001
    n_components = 4
    alpha = 1
    U_64, V_64 = dict_learning_online(X.astype(np.float64), n_components=n_components, max_iter=1000, alpha=alpha, batch_size=10, random_state=0, method=method, tol=0.0, max_no_improvement=None)
    U_32, V_32 = dict_learning_online(X.astype(np.float32), n_components=n_components, max_iter=1000, alpha=alpha, batch_size=10, random_state=0, method=method, tol=0.0, max_no_improvement=None)
    assert_allclose(np.matmul(U_64, V_64), np.matmul(U_32, V_32), rtol=rtol)
    assert_allclose(np.sum(np.abs(U_64)), np.sum(np.abs(U_32)), rtol=rtol)
    assert_allclose(np.sum(V_64 ** 2), np.sum(V_32 ** 2), rtol=rtol)
    assert np.mean(U_64 != 0.0) > 0.05
    assert np.count_nonzero(U_64 != 0.0) == np.count_nonzero(U_32 != 0.0)