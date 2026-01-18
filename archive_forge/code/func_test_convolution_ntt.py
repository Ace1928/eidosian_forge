from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
from sympy.testing.pytest import raises
from sympy.abc import x, y
def test_convolution_ntt():
    p = 7 * 17 * 2 ** 23 + 1
    q = 19 * 2 ** 10 + 1
    r = 2 * 500000003 + 1
    assert all((convolution_ntt([], x, prime=y) == [] for x in ([], [1]) for y in (p, q, r)))
    assert convolution_ntt([2], [3], r) == [6]
    assert convolution_ntt([2, 3], [4], r) == [8, 12]
    assert convolution_ntt([32121, 42144, 4214, 4241], [32132, 3232, 87242], p) == [33867619, 459741727, 79180879, 831885249, 381344700, 369993322]
    assert convolution_ntt([121913, 3171831, 31888131, 12], [17882, 21292, 29921, 312], q) == [8158, 3065, 3682, 7090, 1239, 2232, 3744]
    assert convolution_ntt([12, 19, 21, 98, 67], [2, 6, 7, 8, 9], p) == convolution_ntt([12, 19, 21, 98, 67], [2, 6, 7, 8, 9], q)
    assert convolution_ntt([12, 19, 21, 98, 67], [21, 76, 17, 78, 69], p) == convolution_ntt([12, 19, 21, 98, 67], [21, 76, 17, 78, 69], q)
    raises(ValueError, lambda: convolution_ntt([2, 3], [4, 5], r))
    raises(ValueError, lambda: convolution_ntt([x, y], [y, x], q))
    raises(TypeError, lambda: convolution_ntt(x, y, p))