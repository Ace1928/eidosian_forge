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
def test_continuing(self, *args):
    radius = 0.05
    ns = 6
    engine = self.engine(d=2, radius=radius, scramble=False)
    sample_init = engine.random(n=ns)
    assert len(sample_init) <= ns
    assert l2_norm(sample_init) >= radius
    sample_continued = engine.random(n=ns)
    assert len(sample_continued) <= ns
    assert l2_norm(sample_continued) >= radius
    sample = np.concatenate([sample_init, sample_continued], axis=0)
    assert len(sample) <= ns * 2
    assert l2_norm(sample) >= radius