from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_20news_as_frame(fetch_20newsgroups_vectorized_fxt):
    pd = pytest.importorskip('pandas')
    bunch = fetch_20newsgroups_vectorized_fxt(as_frame=True)
    check_as_frame(bunch, fetch_20newsgroups_vectorized_fxt)
    frame = bunch.frame
    assert frame.shape == (11314, 130108)
    assert all([isinstance(col, pd.SparseDtype) for col in bunch.data.dtypes])
    for expected_feature in ['beginner', 'beginners', 'beginning', 'beginnings', 'begins', 'begley', 'begone']:
        assert expected_feature in frame.keys()
    assert 'category_class' in frame.keys()
    assert bunch.target.name == 'category_class'