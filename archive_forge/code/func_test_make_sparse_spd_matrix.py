import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('norm_diag', [True, False])
@pytest.mark.parametrize('sparse_format', [None, 'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'])
def test_make_sparse_spd_matrix(norm_diag, sparse_format, global_random_seed):
    n_dim = 5
    X = make_sparse_spd_matrix(n_dim=n_dim, norm_diag=norm_diag, sparse_format=sparse_format, random_state=global_random_seed)
    assert X.shape == (n_dim, n_dim), 'X shape mismatch'
    if sparse_format is None:
        assert not sp.issparse(X)
        assert_allclose(X, X.T)
        Xarr = X
    else:
        assert sp.issparse(X) and X.format == sparse_format
        assert_allclose_dense_sparse(X, X.T)
        Xarr = X.toarray()
    from numpy.linalg import eig
    eigenvalues, _ = eig(Xarr)
    assert np.all(eigenvalues > 0), 'X is not positive-definite'
    if norm_diag:
        assert_array_almost_equal(Xarr.diagonal(), np.ones(n_dim))