import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_multilabel_binarizer_given_classes():
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    assert_array_equal(mlb.classes_, [1, 3, 2])
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    assert_array_equal(mlb.classes_, [1, 3, 2])
    mlb = MultiLabelBinarizer(classes=[4, 1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), np.hstack(([[0], [0], [0]], indicator_mat)))
    assert_array_equal(mlb.classes_, [4, 1, 3, 2])
    inp = iter(inp)
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    err_msg = 'The classes argument contains duplicate classes. Remove these duplicates before passing them to MultiLabelBinarizer.'
    mlb = MultiLabelBinarizer(classes=[1, 3, 2, 3])
    with pytest.raises(ValueError, match=err_msg):
        mlb.fit(inp)