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
def test_TransferFunction_construction():
    tf = TransferFunction(s + 1, s ** 2 + s + 1, s)
    assert tf.num == s + 1
    assert tf.den == s ** 2 + s + 1
    assert tf.args == (s + 1, s ** 2 + s + 1, s)
    tf1 = TransferFunction(s + 4, s - 5, s)
    assert tf1.num == s + 4
    assert tf1.den == s - 5
    assert tf1.args == (s + 4, s - 5, s)
    tf2 = TransferFunction(p + 3, p ** 2 - 9, p)
    assert tf2.num == p + 3
    assert tf2.den == p ** 2 - 9
    assert tf2.args == (p + 3, p ** 2 - 9, p)
    tf3 = TransferFunction(p ** 3 + 5 * p ** 2 + 4, p ** 4 + 3 * p + 1, p)
    assert tf3.args == (p ** 3 + 5 * p ** 2 + 4, p ** 4 + 3 * p + 1, p)
    tf4 = TransferFunction((s + 3) * (s - 1), (s - 1) * (s + 5), s)
    assert tf4.den == (s - 1) * (s + 5)
    assert tf4.args == ((s + 3) * (s - 1), (s - 1) * (s + 5), s)
    tf4_ = TransferFunction(p + 2, p + 2, p)
    assert tf4_.args == (p + 2, p + 2, p)
    tf5 = TransferFunction(s - 1, 4 - p, s)
    assert tf5.args == (s - 1, 4 - p, s)
    tf5_ = TransferFunction(s - 1, s - 1, s)
    assert tf5_.args == (s - 1, s - 1, s)
    tf6 = TransferFunction(5, 6, s)
    assert tf6.num == 5
    assert tf6.den == 6
    assert tf6.args == (5, 6, s)
    tf6_ = TransferFunction(1 / 2, 4, s)
    assert tf6_.num == 0.5
    assert tf6_.den == 4
    assert tf6_.args == (0.5, 4, s)
    tf7 = TransferFunction(3 * s ** 2 + 2 * p + 4 * s, 8 * p ** 2 + 7 * s, s)
    tf8 = TransferFunction(3 * s ** 2 + 2 * p + 4 * s, 8 * p ** 2 + 7 * s, p)
    assert not tf7 == tf8
    tf7_ = TransferFunction(a0 * s + a1 * s ** 2 + a2 * s ** 3, b0 * p - b1 * s, s)
    tf8_ = TransferFunction(a0 * s + a1 * s ** 2 + a2 * s ** 3, b0 * p - b1 * s, s)
    assert tf7_ == tf8_
    assert --tf7_ == tf7_ == ----tf7_
    tf9 = TransferFunction(a * s ** 3 + b * s ** 2 + g * s + d, d * p + g * p ** 2 + g * s, s)
    assert tf9.args == (a * s ** 3 + b * s ** 2 + d + g * s, d * p + g * p ** 2 + g * s, s)
    tf10 = TransferFunction(p ** 3 + d, g * s ** 2 + d * s + a, p)
    tf10_ = TransferFunction(p ** 3 + d, g * s ** 2 + d * s + a, p)
    assert tf10.args == (d + p ** 3, a + d * s + g * s ** 2, p)
    assert tf10_ == tf10
    tf11 = TransferFunction(a1 * s + a0, b2 * s ** 2 + b1 * s + b0, s)
    assert tf11.num == a0 + a1 * s
    assert tf11.den == b0 + b1 * s + b2 * s ** 2
    assert tf11.args == (a0 + a1 * s, b0 + b1 * s + b2 * s ** 2, s)
    tf12 = TransferFunction(0, p ** 2 - p + 1, p)
    assert tf12.args == (0, p ** 2 - p + 1, p)
    tf13 = TransferFunction(0, 1, s)
    assert tf13.args == (0, 1, s)
    tf14 = TransferFunction(a0 * s ** 0.5 + a2 * s ** 0.6 - a1, a1 * p ** (-8.7), s)
    assert tf14.args == (a0 * s ** 0.5 - a1 + a2 * s ** 0.6, a1 * p ** (-8.7), s)
    tf15 = TransferFunction(a2 ** 2 * p ** (1 / 4) + a1 * s ** (-4 / 5), a0 * s - p, p)
    assert tf15.args == (a1 * s ** (-0.8) + a2 ** 2 * p ** 0.25, a0 * s - p, p)
    omega_o, k_p, k_o, k_i = symbols('omega_o, k_p, k_o, k_i')
    tf18 = TransferFunction(k_p + k_o * s + k_i / s, s ** 2 + 2 * omega_o * s + omega_o ** 2, s)
    assert tf18.num == k_i / s + k_o * s + k_p
    assert tf18.args == (k_i / s + k_o * s + k_p, omega_o ** 2 + 2 * omega_o * s + s ** 2, s)
    raises(ValueError, lambda: TransferFunction(4, 0, s))
    raises(ValueError, lambda: TransferFunction(s, 0, s))
    raises(ValueError, lambda: TransferFunction(0, 0, s))
    raises(TypeError, lambda: TransferFunction(Matrix([1, 2, 3]), s, s))
    raises(TypeError, lambda: TransferFunction(s ** 2 + 2 * s - 1, s + 3, 3))
    raises(TypeError, lambda: TransferFunction(p + 1, 5 - p, 4))
    raises(TypeError, lambda: TransferFunction(3, 4, 8))