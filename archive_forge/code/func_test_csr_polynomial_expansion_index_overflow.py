import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('degree, n_features', [(2, 65535), (3, 2344), (2, int(np.sqrt(np.iinfo(np.int64).max) + 1)), (3, 65535), (2, int(np.sqrt(np.iinfo(np.int64).max)))])
@pytest.mark.parametrize('interaction_only', [True, False])
@pytest.mark.parametrize('include_bias', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_csr_polynomial_expansion_index_overflow(degree, n_features, interaction_only, include_bias, csr_container):
    """Tests known edge-cases to the dtype promotion strategy and custom
    Cython code, including a current bug in the upstream
    `scipy.sparse.hstack`.
    """
    data = [1.0]
    row = [0]
    col = [n_features - 1]
    expected_indices = [n_features - 1 + int(include_bias)]
    expected_indices.append(n_features * (n_features + 1) // 2 + expected_indices[0])
    expected_indices.append(n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1])
    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(interaction_only=interaction_only, include_bias=include_bias, degree=degree)
    num_combinations = pf._num_combinations(n_features=n_features, min_degree=0, max_degree=degree, interaction_only=pf.interaction_only, include_bias=pf.include_bias)
    if num_combinations > np.iinfo(np.intp).max:
        msg = 'The output that would result from the current configuration would have \\d* features which is too large to be indexed'
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return
    if sp_version < parse_version('1.8.0'):
        has_bug = False
        max_int32 = np.iinfo(np.int32).max
        cumulative_size = n_features + include_bias
        for deg in range(2, degree + 1):
            max_indptr = _calc_total_nnz(X.indptr, interaction_only, deg)
            max_indices = _calc_expanded_nnz(n_features, interaction_only, deg) - 1
            cumulative_size += max_indices + 1
            needs_int64 = max(max_indices, max_indptr) > max_int32
            has_bug |= not needs_int64 and cumulative_size > max_int32
        if has_bug:
            msg = 'In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`'
            with pytest.raises(ValueError, match=msg):
                X_trans = pf.fit_transform(X)
            return
    if sp_version < parse_version('1.9.2') and n_features == 65535 and (degree == 2) and (not interaction_only):
        msg = 'In scipy versions `<1.9.2`, the function `scipy.sparse.hstack`'
        with pytest.raises(ValueError, match=msg):
            X_trans = pf.fit_transform(X)
        return
    X_trans = pf.fit_transform(X)
    expected_dtype = np.int64 if num_combinations > np.iinfo(np.int32).max else np.int32
    non_bias_terms = 1 + (degree - 1) * int(not interaction_only)
    expected_nnz = int(include_bias) + non_bias_terms
    assert X_trans.dtype == X.dtype
    assert X_trans.shape == (1, pf.n_output_features_)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == expected_dtype
    assert X_trans.nnz == expected_nnz
    if include_bias:
        assert X_trans[0, 0] == pytest.approx(1.0)
    for idx in range(non_bias_terms):
        assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)
    offset = interaction_only * n_features
    if degree == 3:
        offset *= 1 + n_features
    assert pf.n_output_features_ == expected_indices[degree - 1] + 1 - offset