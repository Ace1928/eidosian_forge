import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
@pytest.mark.parametrize('dtype', tested_types)
def test_linearoperator_shape_dtype(self, dtype):
    n = 7
    mik = MikotaPair(n, dtype=dtype)
    mik_k = mik.k
    mik_m = mik.m
    assert mik_k.shape == (n, n)
    assert mik_k.dtype == dtype
    assert mik_m.shape == (n, n)
    assert mik_m.dtype == dtype
    mik_default_dtype = MikotaPair(n)
    mikd_k = mik_default_dtype.k
    mikd_m = mik_default_dtype.m
    assert mikd_k.shape == (n, n)
    assert mikd_k.dtype == np.float64
    assert mikd_m.shape == (n, n)
    assert mikd_m.dtype == np.float64
    assert_array_equal(mik_k.toarray(), mikd_k.toarray().astype(dtype))
    assert_array_equal(mik_k.tosparse().toarray(), mikd_k.tosparse().toarray().astype(dtype))