import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_label_binarizer():
    inp = ['pos', 'pos', 'pos', 'pos']
    lb = LabelBinarizer(sparse_output=False)
    expected = np.array([[0, 0, 0, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ['pos'])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)
    lb = LabelBinarizer(sparse_output=True)
    got = lb.fit_transform(inp)
    assert issparse(got)
    assert_array_equal(lb.classes_, ['pos'])
    assert_array_equal(expected, got.toarray())
    assert_array_equal(lb.inverse_transform(got.toarray()), inp)
    lb = LabelBinarizer(sparse_output=False)
    inp = ['neg', 'pos', 'pos', 'neg']
    expected = np.array([[0, 1, 1, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ['neg', 'pos'])
    assert_array_equal(expected, got)
    to_invert = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    assert_array_equal(lb.inverse_transform(to_invert), inp)
    inp = ['spam', 'ham', 'eggs', 'ham', '0']
    expected = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ['0', 'eggs', 'ham', 'spam'])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)