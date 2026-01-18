import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.special._testutils import FuncData
from scipy.special import gamma, gammaln, loggamma
def test_gh_6536():
    z = loggamma(complex(-3.4, +0.0))
    zbar = loggamma(complex(-3.4, -0.0))
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)