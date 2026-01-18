import warnings
import numpy as np
import pytest
from scipy.fft._fftlog import fht, ifht, fhtoffset
from scipy.special import poch
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close
@array_api_compatible
@pytest.mark.parametrize('optimal', [True, False])
@pytest.mark.parametrize('offset', [0.0, 1.0, -1.0])
@pytest.mark.parametrize('bias', [0, 0.1, -0.1])
@pytest.mark.parametrize('n', [64, 63])
def test_fht_identity(n, bias, offset, optimal, xp):
    rng = np.random.RandomState(3491349965)
    a = xp.asarray(rng.standard_normal(n))
    dln = rng.uniform(-1, 1)
    mu = rng.uniform(-2, 2)
    if optimal:
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)
    A = fht(a, dln, mu, offset=offset, bias=bias)
    a_ = ifht(A, dln, mu, offset=offset, bias=bias)
    xp_assert_close(a_, a)