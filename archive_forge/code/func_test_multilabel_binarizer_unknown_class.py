import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_multilabel_binarizer_unknown_class():
    mlb = MultiLabelBinarizer()
    y = [[1, 2]]
    Y = np.array([[1, 0], [0, 1]])
    warning_message = 'unknown class.* will be ignored'
    with pytest.warns(UserWarning, match=warning_message):
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])
    Y = np.array([[1, 0, 0], [0, 1, 0]])
    mlb = MultiLabelBinarizer(classes=[1, 2, 3])
    with pytest.warns(UserWarning, match=warning_message):
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])
    assert_array_equal(matrix, Y)