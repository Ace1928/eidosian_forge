import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_label_binarizer_unseen_labels():
    lb = LabelBinarizer()
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    got = lb.fit_transform(['b', 'd', 'e'])
    assert_array_equal(expected, got)
    expected = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    got = lb.transform(['a', 'b', 'c', 'd', 'e', 'f'])
    assert_array_equal(expected, got)