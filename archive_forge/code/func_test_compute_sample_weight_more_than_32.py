import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_compute_sample_weight_more_than_32():
    y = np.arange(50)
    indices = np.arange(50)
    weight = compute_sample_weight('balanced', y, indices=indices)
    assert_array_almost_equal(weight, np.ones(y.shape[0]))