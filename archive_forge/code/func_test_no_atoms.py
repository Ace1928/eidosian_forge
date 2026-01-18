import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_no_atoms():
    y_empty = np.zeros_like(y)
    Xy_empty = np.dot(X.T, y_empty)
    gamma_empty = ignore_warnings(orthogonal_mp)(X, y_empty, n_nonzero_coefs=1)
    gamma_empty_gram = ignore_warnings(orthogonal_mp)(G, Xy_empty, n_nonzero_coefs=1)
    assert np.all(gamma_empty == 0)
    assert np.all(gamma_empty_gram == 0)