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
def test_MultivariateNormalQMCNonPD(self):
    engine = qmc.MultivariateNormalQMC([0, 0, 0], [[1, 0, 1], [0, 1, 1], [1, 1, 2]])
    assert engine._corr_matrix is not None