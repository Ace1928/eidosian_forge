import mpmath
import random
import pytest
from mpmath import *
def test_jdn():
    """
    Test some special cases of the dn(z, q) function.
    """
    mp.dps = 100
    mstring = str(random.random())
    m = mpf(mstring)
    dn = jdn(zero, m)
    assert dn.ae(one)
    mp.dps = 25
    res = mpf('0.9995017055025556219713297')
    arg = one / 10
    result = jdn(arg, arg)
    assert result.ae(res)
    mp.dps = 15