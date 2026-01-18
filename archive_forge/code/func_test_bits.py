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
@pytest.mark.parametrize('bits', [2, 3])
def test_bits(self, bits):
    engine = qmc.Sobol(2, scramble=False, bits=bits)
    ns = 2 ** bits
    sample = engine.random(ns)
    assert_array_equal(self.unscramble_nd[:ns], sample)
    with pytest.raises(ValueError, match='increasing `bits`'):
        engine.random()