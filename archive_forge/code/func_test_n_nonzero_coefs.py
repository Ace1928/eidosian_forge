import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_n_nonzero_coefs():
    assert np.count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)) <= 5
    assert np.count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5, precompute=True)) <= 5