import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def ellip_harm_known(h2, k2, n, p, s):
    for i in range(h2.size):
        func = known_funcs[int(n[i]), int(p[i])]
        point_ref.append(func(h2[i], k2[i], s[i]))
    return point_ref