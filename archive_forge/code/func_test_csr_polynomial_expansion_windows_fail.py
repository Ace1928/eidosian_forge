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
@pytest.mark.xfail(sys.platform == 'win32', reason='On Windows, scikit-learn is typically compiled with MSVC that does not support int128 arithmetic (at the time of writing)', run=True)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_csr_polynomial_expansion_windows_fail(csr_container):
    n_features = int(np.iinfo(np.int64).max ** (1 / 3) + 3)
    data = [1.0]
    row = [0]
    col = [n_features - 1]
    expected_indices = [n_features - 1]
    expected_indices.append(int(n_features * (n_features + 1) // 2 + expected_indices[0]))
    expected_indices.append(int(n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1]))
    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(interaction_only=False, include_bias=False, degree=3)
    if sys.maxsize <= 2 ** 32:
        msg = 'The output that would result from the current configuration would have \\d* features which is too large to be indexed'
        with pytest.raises(ValueError, match=msg):
            pf.fit_transform(X)
    else:
        X_trans = pf.fit_transform(X)
        for idx in range(3):
            assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)