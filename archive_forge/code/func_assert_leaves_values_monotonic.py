import re
import numpy as np
import pytest
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
def assert_leaves_values_monotonic(predictor, monotonic_cst):
    nodes = predictor.nodes

    def get_leaves_values():
        """get leaves values from left to right"""
        values = []

        def depth_first_collect_leaf_values(node_idx):
            node = nodes[node_idx]
            if node['is_leaf']:
                values.append(node['value'])
                return
            depth_first_collect_leaf_values(node['left'])
            depth_first_collect_leaf_values(node['right'])
        depth_first_collect_leaf_values(0)
        return values
    values = get_leaves_values()
    if monotonic_cst == MonotonicConstraint.NO_CST:
        assert not is_increasing(values) and (not is_decreasing(values))
    elif monotonic_cst == MonotonicConstraint.POS:
        assert is_increasing(values)
    else:
        assert is_decreasing(values)