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
@pytest.mark.parametrize('interaction_only', [True, False])
@pytest.mark.parametrize('include_bias', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_csr_polynomial_expansion_too_large_to_index(interaction_only, include_bias, csr_container):
    n_features = np.iinfo(np.int64).max // 2
    data = [1.0]
    row = [0]
    col = [n_features - 1]
    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(interaction_only=interaction_only, include_bias=include_bias, degree=(2, 2))
    msg = 'The output that would result from the current configuration would have \\d* features which is too large to be indexed'
    with pytest.raises(ValueError, match=msg):
        pf.fit(X)
    with pytest.raises(ValueError, match=msg):
        pf.fit_transform(X)