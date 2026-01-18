from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_20news(fetch_20newsgroups_fxt):
    data = fetch_20newsgroups_fxt(subset='all', shuffle=False)
    assert data.DESCR.startswith('.. _20newsgroups_dataset:')
    data2cats = fetch_20newsgroups_fxt(subset='all', categories=data.target_names[-1:-3:-1], shuffle=False)
    assert data2cats.target_names == data.target_names[-2:]
    assert np.unique(data2cats.target).tolist() == [0, 1]
    assert len(data2cats.filenames) == len(data2cats.target)
    assert len(data2cats.filenames) == len(data2cats.data)
    entry1 = data2cats.data[0]
    category = data2cats.target_names[data2cats.target[0]]
    label = data.target_names.index(category)
    entry2 = data.data[np.where(data.target == label)[0][0]]
    assert entry1 == entry2
    X, y = fetch_20newsgroups_fxt(subset='all', shuffle=False, return_X_y=True)
    assert len(X) == len(data.data)
    assert y.shape == data.target.shape