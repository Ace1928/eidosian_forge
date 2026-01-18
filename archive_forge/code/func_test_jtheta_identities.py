import mpmath
import random
import pytest
from mpmath import *
def test_jtheta_identities():
    """
    Tests the some of the jacobi identidies found in Abramowitz,
    Sec. 16.28, Pg. 576. The identities are tested to 1 part in 10^98.
    """
    mp.dps = 110
    eps1 = ldexp(eps, 30)
    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)
        zstring = str(10 * random.random())
        z = mpf(zstring)
        term1 = jtheta(1, z, q) ** 2 * jtheta(4, zero, q) ** 2
        term2 = jtheta(3, z, q) ** 2 * jtheta(2, zero, q) ** 2
        term3 = jtheta(2, z, q) ** 2 * jtheta(3, zero, q) ** 2
        equality = term1 - term2 + term3
        assert equality.ae(0, eps1)
        zstring = str(100 * random.random())
        z = mpf(zstring)
        term1 = jtheta(2, z, q) ** 2 * jtheta(4, zero, q) ** 2
        term2 = jtheta(4, z, q) ** 2 * jtheta(2, zero, q) ** 2
        term3 = jtheta(1, z, q) ** 2 * jtheta(3, zero, q) ** 2
        equality = term1 - term2 + term3
        assert equality.ae(0, eps1)
        term1 = jtheta(3, z, q) ** 2 * jtheta(4, zero, q) ** 2
        term2 = jtheta(4, z, q) ** 2 * jtheta(3, zero, q) ** 2
        term3 = jtheta(1, z, q) ** 2 * jtheta(2, zero, q) ** 2
        equality = term1 - term2 + term3
        assert equality.ae(0, eps1)
        term1 = jtheta(4, z, q) ** 2 * jtheta(4, zero, q) ** 2
        term2 = jtheta(3, z, q) ** 2 * jtheta(3, zero, q) ** 2
        term3 = jtheta(2, z, q) ** 2 * jtheta(2, zero, q) ** 2
        equality = term1 - term2 + term3
        assert equality.ae(0, eps1)
        term1 = jtheta(2, zero, q) ** 4
        term2 = jtheta(4, zero, q) ** 4
        term3 = jtheta(3, zero, q) ** 4
        equality = term1 + term2 - term3
        assert equality.ae(0, eps1)
    mp.dps = 15