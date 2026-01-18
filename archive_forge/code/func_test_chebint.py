from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebint(self):
    assert_raises(TypeError, cheb.chebint, [0], 0.5)
    assert_raises(ValueError, cheb.chebint, [0], -1)
    assert_raises(ValueError, cheb.chebint, [0], 1, [0, 0])
    assert_raises(ValueError, cheb.chebint, [0], lbnd=[0])
    assert_raises(ValueError, cheb.chebint, [0], scl=[0])
    assert_raises(TypeError, cheb.chebint, [0], axis=0.5)
    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = cheb.chebint([0], m=i, k=k)
        assert_almost_equal(res, [0, 1])
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        chebpol = cheb.poly2cheb(pol)
        chebint = cheb.chebint(chebpol, m=1, k=[i])
        res = cheb.cheb2poly(chebint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        chebpol = cheb.poly2cheb(pol)
        chebint = cheb.chebint(chebpol, m=1, k=[i], lbnd=-1)
        assert_almost_equal(cheb.chebval(-1, chebint), i)
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        chebpol = cheb.poly2cheb(pol)
        chebint = cheb.chebint(chebpol, m=1, k=[i], scl=2)
        res = cheb.cheb2poly(chebint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = cheb.chebint(tgt, m=1)
            res = cheb.chebint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = cheb.chebint(tgt, m=1, k=[k])
            res = cheb.chebint(pol, m=j, k=list(range(j)))
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = cheb.chebint(tgt, m=1, k=[k], lbnd=-1)
            res = cheb.chebint(pol, m=j, k=list(range(j)), lbnd=-1)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = cheb.chebint(tgt, m=1, k=[k], scl=2)
            res = cheb.chebint(pol, m=j, k=list(range(j)), scl=2)
            assert_almost_equal(trim(res), trim(tgt))