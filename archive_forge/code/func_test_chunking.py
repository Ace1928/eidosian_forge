import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
@pytest.mark.slow
def test_chunking(self, monkeypatch):
    rng = np.random.RandomState(0)
    seq = Halton(2, scramble=False, seed=rng)
    degree = 3
    largeN = 1000 + 33
    x = seq.random(50)
    xitp = seq.random(largeN)
    P = _vandermonde(x, degree)
    Pitp = _vandermonde(xitp, degree)
    poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
    y = P.dot(poly_coeffs)
    yitp1 = Pitp.dot(poly_coeffs)
    interp = self.build(x, y, degree=degree)
    ce_real = interp._chunk_evaluator

    def _chunk_evaluator(*args, **kwargs):
        kwargs.update(memory_budget=100)
        return ce_real(*args, **kwargs)
    monkeypatch.setattr(interp, '_chunk_evaluator', _chunk_evaluator)
    yitp2 = interp(xitp)
    assert_allclose(yitp1, yitp2, atol=1e-08)