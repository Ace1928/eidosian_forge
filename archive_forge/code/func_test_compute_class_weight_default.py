import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_compute_class_weight_default():
    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)
    classes_len = len(classes)
    cw = compute_class_weight(None, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, np.ones(3))
    cw = compute_class_weight({2: 1.5}, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, [1.5, 1.0, 1.0])
    cw = compute_class_weight({2: 1.5, 4: 0.5}, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, [1.5, 1.0, 0.5])