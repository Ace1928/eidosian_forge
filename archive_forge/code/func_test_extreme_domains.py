import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_extreme_domains(self, kernel):
    seq = Halton(2, scramble=False, seed=np.random.RandomState())
    scale = 1e+50
    shift = 1e+55
    x = seq.random(100)
    y = _2d_test_function(x)
    xitp = seq.random(100)
    if kernel in _SCALE_INVARIANT:
        yitp1 = self.build(x, y, kernel=kernel)(xitp)
        yitp2 = self.build(x * scale + shift, y, kernel=kernel)(xitp * scale + shift)
    else:
        yitp1 = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
        yitp2 = self.build(x * scale + shift, y, epsilon=5.0 / scale, kernel=kernel)(xitp * scale + shift)
    assert_allclose(yitp1, yitp2, atol=1e-08)