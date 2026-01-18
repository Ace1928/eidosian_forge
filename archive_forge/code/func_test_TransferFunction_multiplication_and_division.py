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
def test_TransferFunction_multiplication_and_division():
    G1 = TransferFunction(s + 3, -s ** 3 + 9, s)
    G2 = TransferFunction(s + 1, s - 5, s)
    G3 = TransferFunction(p, p ** 4 - 6, p)
    G4 = TransferFunction(p + 4, p - 5, p)
    G5 = TransferFunction(s + 6, s - 5, s)
    G6 = TransferFunction(s + 3, s + 1, s)
    G7 = TransferFunction(1, 1, s)
    assert G1 * G2 == Series(G1, G2)
    assert -G1 * G5 == Series(-G1, G5)
    assert -G2 * G5 * -G6 == Series(-G2, G5, -G6)
    assert -G1 * -G2 * -G5 * -G6 == Series(-G1, -G2, -G5, -G6)
    assert G3 * G4 == Series(G3, G4)
    assert G1 * G2 * -(G5 * G6) == Series(G1, G2, TransferFunction(-1, 1, s), Series(G5, G6))
    assert G1 * G2 * (G5 + G6) == Series(G1, G2, Parallel(G5, G6))
    c = symbols('c', commutative=False)
    raises(ValueError, lambda: G3 * Matrix([1, 2, 3]))
    raises(ValueError, lambda: G1 * c)
    raises(ValueError, lambda: G3 * G5)
    raises(ValueError, lambda: G5 * (s - 1))
    raises(ValueError, lambda: 9 * G5)
    raises(ValueError, lambda: G3 / Matrix([1, 2, 3]))
    raises(ValueError, lambda: G6 / 0)
    raises(ValueError, lambda: G3 / G5)
    raises(ValueError, lambda: G5 / 2)
    raises(ValueError, lambda: G5 / s ** 2)
    raises(ValueError, lambda: (s - 4 * s ** 2) / G2)
    raises(ValueError, lambda: 0 / G4)
    raises(ValueError, lambda: G5 / G6)
    raises(ValueError, lambda: -G3 / G4)
    raises(ValueError, lambda: G7 / (1 + G6))
    raises(ValueError, lambda: G7 / (G5 * G6))
    raises(ValueError, lambda: G7 / (G7 + (G5 + G6)))