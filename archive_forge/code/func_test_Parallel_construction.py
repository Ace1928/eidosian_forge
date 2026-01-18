from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix
from sympy.physics.control import (TransferFunction, Series, Parallel,
from sympy.testing.pytest import raises
def test_Parallel_construction():
    tf = TransferFunction(a0 * s ** 3 + a1 * s ** 2 - a2 * s, b0 * p ** 4 + b1 * p ** 3 - b2 * s * p, s)
    tf2 = TransferFunction(a2 * p - s, a2 * s + p, s)
    tf3 = TransferFunction(a0 * p + p ** a1 - s, p, p)
    tf4 = TransferFunction(1, s ** 2 + 2 * zeta * wn * s + wn ** 2, s)
    inp = Function('X_d')(s)
    out = Function('X')(s)
    p0 = Parallel(tf, tf2)
    assert p0.args == (tf, tf2)
    assert p0.var == s
    p1 = Parallel(Series(tf, -tf2), tf2)
    assert p1.args == (Series(tf, -tf2), tf2)
    assert p1.var == s
    tf3_ = TransferFunction(inp, 1, s)
    tf4_ = TransferFunction(-out, 1, s)
    p2 = Parallel(tf, Series(tf3_, -tf4_), tf2)
    assert p2.args == (tf, Series(tf3_, -tf4_), tf2)
    p3 = Parallel(tf, tf2, tf4)
    assert p3.args == (tf, tf2, tf4)
    p4 = Parallel(tf3_, tf4_)
    assert p4.args == (tf3_, tf4_)
    assert p4.var == s
    p5 = Parallel(tf, tf2)
    assert p0 == p5
    assert not p0 == p1
    p6 = Parallel(tf2, tf4, Series(tf2, -tf4))
    assert p6.args == (tf2, tf4, Series(tf2, -tf4))
    p7 = Parallel(tf2, tf4, Series(tf2, -tf), tf4)
    assert p7.args == (tf2, tf4, Series(tf2, -tf), tf4)
    raises(ValueError, lambda: Parallel(tf, tf3))
    raises(ValueError, lambda: Parallel(tf, tf2, tf3, tf4))
    raises(ValueError, lambda: Parallel(-tf3, tf4))
    raises(TypeError, lambda: Parallel(2, tf, tf4))
    raises(TypeError, lambda: Parallel(s ** 2 + p * s, tf3, tf2))
    raises(TypeError, lambda: Parallel(tf3, Matrix([1, 2, 3, 4])))