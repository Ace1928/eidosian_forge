import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_label_binarizer_set_label_encoding():
    lb = LabelBinarizer(neg_label=-2, pos_label=0)
    inp = np.array([0, 1, 1, 0])
    expected = np.array([[-2, 0, 0, -2]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)
    lb = LabelBinarizer(neg_label=-2, pos_label=2)
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array([[-2, -2, -2, +2], [-2, -2, +2, -2], [-2, +2, -2, -2], [-2, -2, +2, -2], [+2, -2, -2, -2]])
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)