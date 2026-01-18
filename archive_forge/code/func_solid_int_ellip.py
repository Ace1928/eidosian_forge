import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def solid_int_ellip(lambda1, mu, nu, n, p, h2, k2):
    return ellip_harm(h2, k2, n, p, lambda1) * ellip_harm(h2, k2, n, p, mu) * ellip_harm(h2, k2, n, p, nu)