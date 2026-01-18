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
@pytest.mark.parametrize('X, Y', [(X64, Y64), (X32, Y32), (X_mmap, Y_mmap)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_haversine_metric(X, Y, csr_container):
    X = np.asarray(X[:, :2])
    Y = np.asarray(Y[:, :2])
    X_csr, Y_csr = (csr_container(X), csr_container(Y))

    def haversine_slow(x1, x2):
        return 2 * np.arcsin(np.sqrt(np.sin(0.5 * (x1[0] - x2[0])) ** 2 + np.cos(x1[0]) * np.cos(x2[0]) * np.sin(0.5 * (x1[1] - x2[1])) ** 2))
    D_reference = np.zeros((X_csr.shape[0], Y_csr.shape[0]))
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            D_reference[i, j] = haversine_slow(xi, yj)
    haversine = DistanceMetric.get_metric('haversine', X.dtype)
    D_sklearn = haversine.pairwise(X, Y)
    assert_allclose(haversine.dist_to_rdist(D_sklearn), np.sin(0.5 * D_reference) ** 2, rtol=1e-06)
    assert_allclose(D_sklearn, D_reference)
    D_sklearn = haversine.pairwise(X_csr, Y_csr)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_reference)
    D_sklearn = haversine.pairwise(X_csr, Y)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_reference)
    D_sklearn = haversine.pairwise(X, Y_csr)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_reference)