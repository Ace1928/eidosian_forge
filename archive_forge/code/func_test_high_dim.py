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
def test_high_dim(self):
    engine = qmc.Sobol(1111, scramble=False)
    count1 = Counter(engine.random().flatten().tolist())
    count2 = Counter(engine.random().flatten().tolist())
    assert_equal(count1, Counter({0.0: 1111}))
    assert_equal(count2, Counter({0.5: 1111}))