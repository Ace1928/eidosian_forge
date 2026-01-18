from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_20news_normalization(fetch_20newsgroups_vectorized_fxt):
    X = fetch_20newsgroups_vectorized_fxt(normalize=False)
    X_ = fetch_20newsgroups_vectorized_fxt(normalize=True)
    X_norm = X_['data'][:100]
    X = X['data'][:100]
    assert_allclose_dense_sparse(X_norm, normalize(X))
    assert np.allclose(np.linalg.norm(X_norm.todense(), axis=1), 1)