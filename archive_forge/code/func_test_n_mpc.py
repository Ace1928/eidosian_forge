from symengine.test_utilities import raises
from symengine import (Symbol, sin, cos, Integer, Add, I, RealDouble, ComplexDouble, sqrt)
from unittest.case import SkipTest
def test_n_mpc():
    x = sqrt(Integer(2)) + 3 * I
    try:
        from symengine import ComplexMPC
        y = ComplexMPC('1.41421356237309504880169', '3.0', 75)
        assert x.n(75) == y
    except ImportError:
        raises(Exception, lambda: x.n(75, real=True))
        raises(ValueError, lambda: x.n(75, real=False))
        raises(ValueError, lambda: x.n(75))
        raise SkipTest('No MPC support')