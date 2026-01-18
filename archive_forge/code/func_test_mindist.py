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
def test_mindist(self):
    rng = np.random.default_rng(132074951149370773672162394161442690287)
    ns = 50
    low, high = (0.08, 0.2)
    radii = (high - low) * rng.random(5) + low
    dimensions = [1, 3, 4]
    hypersphere_methods = ['volume', 'surface']
    gen = product(dimensions, radii, hypersphere_methods)
    for d, radius, hypersphere in gen:
        engine = self.qmce(d=d, radius=radius, hypersphere=hypersphere, seed=rng)
        sample = engine.random(ns)
        assert len(sample) <= ns
        assert l2_norm(sample) >= radius