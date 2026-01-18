import numpy as np
import pytest
import scipy.sparse as sp
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import linalg
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.sparsefuncs import (
from sklearn.utils.sparsefuncs_fast import (
def test_inplace_column_scale():
    rng = np.random.RandomState(0)
    X = sp.rand(100, 200, 0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(200)
    XA *= scale
    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)
    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale
    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    with pytest.raises(TypeError):
        inplace_column_scale(X.tolil(), scale)