import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_lloyd_errors(self):
    with pytest.raises(ValueError, match='`sample` is not a 2D array'):
        sample = [0, 1, 0.5]
        _lloyd_centroidal_voronoi_tessellation(sample)
    msg = '`sample` dimension is not >= 2'
    with pytest.raises(ValueError, match=msg):
        sample = [[0], [0.4], [1]]
        _lloyd_centroidal_voronoi_tessellation(sample)
    msg = '`sample` is not in unit hypercube'
    with pytest.raises(ValueError, match=msg):
        sample = [[-1.1, 0], [0.1, 0.4], [1, 2]]
        _lloyd_centroidal_voronoi_tessellation(sample)