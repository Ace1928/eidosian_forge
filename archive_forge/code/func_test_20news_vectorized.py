from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_20news_vectorized(fetch_20newsgroups_vectorized_fxt):
    bunch = fetch_20newsgroups_vectorized_fxt(subset='train')
    assert sp.issparse(bunch.data) and bunch.data.format == 'csr'
    assert bunch.data.shape == (11314, 130107)
    assert bunch.target.shape[0] == 11314
    assert bunch.data.dtype == np.float64
    assert bunch.DESCR.startswith('.. _20newsgroups_dataset:')
    bunch = fetch_20newsgroups_vectorized_fxt(subset='test')
    assert sp.issparse(bunch.data) and bunch.data.format == 'csr'
    assert bunch.data.shape == (7532, 130107)
    assert bunch.target.shape[0] == 7532
    assert bunch.data.dtype == np.float64
    assert bunch.DESCR.startswith('.. _20newsgroups_dataset:')
    fetch_func = partial(fetch_20newsgroups_vectorized_fxt, subset='test')
    check_return_X_y(bunch, fetch_func)
    bunch = fetch_20newsgroups_vectorized_fxt(subset='all')
    assert sp.issparse(bunch.data) and bunch.data.format == 'csr'
    assert bunch.data.shape == (11314 + 7532, 130107)
    assert bunch.target.shape[0] == 11314 + 7532
    assert bunch.data.dtype == np.float64
    assert bunch.DESCR.startswith('.. _20newsgroups_dataset:')