import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
def test_get_support():
    sel = StepSelector()
    sel.fit(X, y)
    assert_array_equal(support, sel.get_support())
    assert_array_equal(support_inds, sel.get_support(indices=True))