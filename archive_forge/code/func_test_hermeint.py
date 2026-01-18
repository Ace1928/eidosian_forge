from functools import reduce
import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermeint(self):
    assert_raises(TypeError, herme.hermeint, [0], 0.5)
    assert_raises(ValueError, herme.hermeint, [0], -1)
    assert_raises(ValueError, herme.hermeint, [0], 1, [0, 0])
    assert_raises(ValueError, herme.hermeint, [0], lbnd=[0])
    assert_raises(ValueError, herme.hermeint, [0], scl=[0])
    assert_raises(TypeError, herme.hermeint, [0], axis=0.5)
    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = herme.hermeint([0], m=i, k=k)
        assert_almost_equal(res, [0, 1])
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermepol = herme.poly2herme(pol)
        hermeint = herme.hermeint(hermepol, m=1, k=[i])
        res = herme.herme2poly(hermeint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermepol = herme.poly2herme(pol)
        hermeint = herme.hermeint(hermepol, m=1, k=[i], lbnd=-1)
        assert_almost_equal(herme.hermeval(-1, hermeint), i)
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermepol = herme.poly2herme(pol)
        hermeint = herme.hermeint(hermepol, m=1, k=[i], scl=2)
        res = herme.herme2poly(hermeint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herme.hermeint(tgt, m=1)
            res = herme.hermeint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herme.hermeint(tgt, m=1, k=[k])
            res = herme.hermeint(pol, m=j, k=list(range(j)))
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herme.hermeint(tgt, m=1, k=[k], lbnd=-1)
            res = herme.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herme.hermeint(tgt, m=1, k=[k], scl=2)
            res = herme.hermeint(pol, m=j, k=list(range(j)), scl=2)
            assert_almost_equal(trim(res), trim(tgt))