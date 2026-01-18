from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_20news_length_consistency(fetch_20newsgroups_fxt):
    """Checks the length consistencies within the bunch

    This is a non-regression test for a bug present in 0.16.1.
    """
    data = fetch_20newsgroups_fxt(subset='all')
    assert len(data['data']) == len(data.data)
    assert len(data['target']) == len(data.target)
    assert len(data['filenames']) == len(data.filenames)