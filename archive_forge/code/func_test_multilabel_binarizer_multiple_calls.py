import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_multilabel_binarizer_multiple_calls():
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    indicator_mat2 = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    mlb.classes = [1, 2, 3]
    assert_array_equal(mlb.fit_transform(inp), indicator_mat2)