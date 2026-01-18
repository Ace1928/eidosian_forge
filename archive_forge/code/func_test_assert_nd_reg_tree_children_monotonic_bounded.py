import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
def test_assert_nd_reg_tree_children_monotonic_bounded():
    X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
    y = np.sin(X).ravel()
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)
    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])
    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])
    assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [0])
    X = np.linspace(-5, 5, 5).reshape(-1, 1)
    y = X.ravel() ** 3
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)
    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, -y)
    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])