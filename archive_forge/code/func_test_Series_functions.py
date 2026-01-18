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
def test_Series_functions():
    tf1 = TransferFunction(1, s ** 2 + 2 * zeta * wn * s + wn ** 2, s)
    tf2 = TransferFunction(k, 1, s)
    tf3 = TransferFunction(a2 * p - s, a2 * s + p, s)
    tf4 = TransferFunction(a0 * p + p ** a1 - s, p, p)
    tf5 = TransferFunction(a1 * s ** 2 + a2 * s - a0, s + a0, s)
    assert tf1 * tf2 * tf3 == Series(tf1, tf2, tf3) == Series(Series(tf1, tf2), tf3) == Series(tf1, Series(tf2, tf3))
    assert tf1 * (tf2 + tf3) == Series(tf1, Parallel(tf2, tf3))
    assert tf1 * tf2 + tf5 == Parallel(Series(tf1, tf2), tf5)
    assert tf1 * tf2 - tf5 == Parallel(Series(tf1, tf2), -tf5)
    assert tf1 * tf2 + tf3 + tf5 == Parallel(Series(tf1, tf2), tf3, tf5)
    assert tf1 * tf2 - tf3 - tf5 == Parallel(Series(tf1, tf2), -tf3, -tf5)
    assert tf1 * tf2 - tf3 + tf5 == Parallel(Series(tf1, tf2), -tf3, tf5)
    assert tf1 * tf2 + tf3 * tf5 == Parallel(Series(tf1, tf2), Series(tf3, tf5))
    assert tf1 * tf2 - tf3 * tf5 == Parallel(Series(tf1, tf2), Series(TransferFunction(-1, 1, s), Series(tf3, tf5)))
    assert tf2 * tf3 * (tf2 - tf1) * tf3 == Series(tf2, tf3, Parallel(tf2, -tf1), tf3)
    assert -tf1 * tf2 == Series(-tf1, tf2)
    assert -(tf1 * tf2) == Series(TransferFunction(-1, 1, s), Series(tf1, tf2))
    raises(ValueError, lambda: tf1 * tf2 * tf4)
    raises(ValueError, lambda: tf1 * (tf2 - tf4))
    raises(ValueError, lambda: tf3 * Matrix([1, 2, 3]))
    assert Series(tf1, tf2, evaluate=True) == Series(tf1, tf2).doit() == TransferFunction(k, s ** 2 + 2 * s * wn * zeta + wn ** 2, s)
    assert Series(tf1, tf2, Parallel(tf1, -tf3), evaluate=True) == Series(tf1, tf2, Parallel(tf1, -tf3)).doit() == TransferFunction(k * (a2 * s + p + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2, s)
    assert Series(tf2, tf1, -tf3, evaluate=True) == Series(tf2, tf1, -tf3).doit() == TransferFunction(k * (-a2 * p + s), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert not Series(tf1, -tf2, evaluate=False) == Series(tf1, -tf2).doit()
    assert Series(Parallel(tf1, tf2), Parallel(tf2, -tf3)).doit() == TransferFunction((k * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + 1) * (-a2 * p + k * (a2 * s + p) + s), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Series(-tf1, -tf2, -tf3).doit() == TransferFunction(k * (-a2 * p + s), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert -Series(tf1, tf2, tf3).doit() == TransferFunction(-k * (a2 * p - s), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Series(tf2, tf3, Parallel(tf2, -tf1), tf3).doit() == TransferFunction(k * (a2 * p - s) ** 2 * (k * (s ** 2 + 2 * s * wn * zeta + wn ** 2) - 1), (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    assert Series(tf1, tf2).rewrite(TransferFunction) == TransferFunction(k, s ** 2 + 2 * s * wn * zeta + wn ** 2, s)
    assert Series(tf2, tf1, -tf3).rewrite(TransferFunction) == TransferFunction(k * (-a2 * p + s), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s)
    S1 = Series(Parallel(tf1, tf2), Parallel(tf2, -tf3))
    assert S1.is_proper
    assert not S1.is_strictly_proper
    assert S1.is_biproper
    S2 = Series(tf1, tf2, tf3)
    assert S2.is_proper
    assert S2.is_strictly_proper
    assert not S2.is_biproper
    S3 = Series(tf1, -tf2, Parallel(tf1, -tf3))
    assert S3.is_proper
    assert S3.is_strictly_proper
    assert not S3.is_biproper