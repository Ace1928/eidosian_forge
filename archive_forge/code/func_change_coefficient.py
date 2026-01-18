import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def change_coefficient(lambda1, mu, nu, h2, k2):
    x = sqrt(lambda1 ** 2 * mu ** 2 * nu ** 2 / (h2 * k2))
    y = sqrt((lambda1 ** 2 - h2) * (mu ** 2 - h2) * (h2 - nu ** 2) / (h2 * (k2 - h2)))
    z = sqrt((lambda1 ** 2 - k2) * (k2 - mu ** 2) * (k2 - nu ** 2) / (k2 * (k2 - h2)))
    return (x, y, z)