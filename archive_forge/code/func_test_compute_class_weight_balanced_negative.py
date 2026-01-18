import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_compute_class_weight_balanced_negative():
    classes = np.array([-2, -1, 0])
    y = np.asarray([-1, -1, 0, 0, -2, -2])
    cw = compute_class_weight('balanced', classes=classes, y=y)
    assert len(cw) == len(classes)
    assert_array_almost_equal(cw, np.array([1.0, 1.0, 1.0]))
    y = np.asarray([-1, 0, 0, -2, -2, -2])
    cw = compute_class_weight('balanced', classes=classes, y=y)
    assert len(cw) == len(classes)
    class_counts = np.bincount(y + 2)
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0 / 3, 2.0, 1.0])