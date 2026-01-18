import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
def test_array_with_nans_fails(self):
    points_with_nan = np.array([(0, 0), (1, 1), (2, np.nan)], dtype=np.float64)
    assert_raises(ValueError, qhull.ConvexHull, points_with_nan)