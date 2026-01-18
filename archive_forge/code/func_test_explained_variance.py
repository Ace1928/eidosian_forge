import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
@pytest.mark.parametrize('kind', ('dense', 'sparse'))
@pytest.mark.parametrize('n_components', [10, 20])
@pytest.mark.parametrize('solver', SVD_SOLVERS)
def test_explained_variance(X_sparse, kind, n_components, solver):
    X = X_sparse if kind == 'sparse' else X_sparse.toarray()
    svd = TruncatedSVD(n_components, algorithm=solver)
    X_tr = svd.fit_transform(X)
    assert_array_less(0.0, svd.explained_variance_ratio_)
    assert_array_less(svd.explained_variance_ratio_.sum(), 1.0)
    total_variance = np.var(X_sparse.toarray(), axis=0).sum()
    variances = np.var(X_tr, axis=0)
    true_explained_variance_ratio = variances / total_variance
    assert_allclose(svd.explained_variance_ratio_, true_explained_variance_ratio)