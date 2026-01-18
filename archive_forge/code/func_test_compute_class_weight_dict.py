import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_compute_class_weight_dict():
    classes = np.arange(3)
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0}
    y = np.asarray([0, 0, 1, 2])
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_array_almost_equal(np.asarray([1.0, 2.0, 3.0]), cw)
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0, 4: 1.5}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([1.0, 2.0, 3.0], cw)
    class_weights = {-1: 5.0, 0: 4.0, 1: 2.0, 2: 3.0}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([4.0, 2.0, 3.0], cw)