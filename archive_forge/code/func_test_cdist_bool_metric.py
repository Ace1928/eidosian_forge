import copy
import itertools
import pickle
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, create_memmap_backed_data
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version, sp_version
@pytest.mark.parametrize('metric', BOOL_METRICS)
@pytest.mark.parametrize('X_bool, Y_bool', [(X_bool, Y_bool), (X_bool_mmap, Y_bool_mmap)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_cdist_bool_metric(metric, X_bool, Y_bool, csr_container):
    D_scipy_cdist = cdist(X_bool, Y_bool, metric)
    dm = DistanceMetric.get_metric(metric)
    D_sklearn = dm.pairwise(X_bool, Y_bool)
    assert_allclose(D_sklearn, D_scipy_cdist)
    X_bool_csr, Y_bool_csr = (csr_container(X_bool), csr_container(Y_bool))
    D_sklearn = dm.pairwise(X_bool, Y_bool)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_scipy_cdist)
    D_sklearn = dm.pairwise(X_bool_csr, Y_bool_csr)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_scipy_cdist)
    D_sklearn = dm.pairwise(X_bool, Y_bool_csr)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_scipy_cdist)
    D_sklearn = dm.pairwise(X_bool_csr, Y_bool)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_scipy_cdist)