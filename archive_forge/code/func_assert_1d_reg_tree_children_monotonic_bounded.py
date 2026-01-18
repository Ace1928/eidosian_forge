import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
def assert_1d_reg_tree_children_monotonic_bounded(tree_, monotonic_sign):
    values = tree_.value
    for i in range(tree_.node_count):
        if tree_.children_left[i] > i and tree_.children_right[i] > i:
            i_left = tree_.children_left[i]
            i_right = tree_.children_right[i]
            if monotonic_sign == 1:
                assert values[i_left] <= values[i_right]
            elif monotonic_sign == -1:
                assert values[i_left] >= values[i_right]
            val_middle = (values[i_left] + values[i_right]) / 2
            if tree_.feature[i_left] >= 0:
                i_left_right = tree_.children_right[i_left]
                if monotonic_sign == 1:
                    assert values[i_left_right] <= val_middle
                elif monotonic_sign == -1:
                    assert values[i_left_right] >= val_middle
            if tree_.feature[i_right] >= 0:
                i_right_left = tree_.children_left[i_right]
                if monotonic_sign == 1:
                    assert val_middle <= values[i_right_left]
                elif monotonic_sign == -1:
                    assert val_middle >= values[i_right_left]