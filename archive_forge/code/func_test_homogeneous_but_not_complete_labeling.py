import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_homogeneous_but_not_complete_labeling():
    h, c, v = homogeneity_completeness_v_measure([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 2, 2])
    assert_almost_equal(h, 1.0, 2)
    assert_almost_equal(c, 0.69, 2)
    assert_almost_equal(v, 0.81, 2)