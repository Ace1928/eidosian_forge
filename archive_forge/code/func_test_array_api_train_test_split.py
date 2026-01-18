import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('array_namespace, device, dtype_name', yield_namespace_device_dtype_combinations())
@pytest.mark.parametrize('shuffle,stratify', ((True, None), (True, np.hstack((np.ones(6), np.zeros(4)))), (False, None)))
def test_array_api_train_test_split(shuffle, stratify, array_namespace, device, dtype_name):
    xp = _array_api_for_tests(array_namespace, device)
    X = np.arange(100).reshape((10, 10))
    y = np.arange(10)
    X_np = X.astype(dtype_name)
    X_xp = xp.asarray(X_np, device=device)
    y_np = y.astype(dtype_name)
    y_xp = xp.asarray(y_np, device=device)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y, random_state=0, shuffle=shuffle, stratify=stratify)
    with config_context(array_api_dispatch=True):
        if stratify is not None:
            stratify_xp = xp.asarray(stratify)
        else:
            stratify_xp = stratify
        X_train_xp, X_test_xp, y_train_xp, y_test_xp = train_test_split(X_xp, y_xp, shuffle=shuffle, stratify=stratify_xp, random_state=0)
        assert get_namespace(X_train_xp)[0] == get_namespace(X_xp)[0]
        assert get_namespace(X_test_xp)[0] == get_namespace(X_xp)[0]
        assert get_namespace(y_train_xp)[0] == get_namespace(y_xp)[0]
        assert get_namespace(y_test_xp)[0] == get_namespace(y_xp)[0]
    assert array_api_device(X_train_xp) == array_api_device(X_xp)
    assert array_api_device(y_train_xp) == array_api_device(y_xp)
    assert array_api_device(X_test_xp) == array_api_device(X_xp)
    assert array_api_device(y_test_xp) == array_api_device(y_xp)
    assert X_train_xp.dtype == X_xp.dtype
    assert y_train_xp.dtype == y_xp.dtype
    assert X_test_xp.dtype == X_xp.dtype
    assert y_test_xp.dtype == y_xp.dtype
    assert_allclose(_convert_to_numpy(X_train_xp, xp=xp), X_train_np)
    assert_allclose(_convert_to_numpy(X_test_xp, xp=xp), X_test_np)