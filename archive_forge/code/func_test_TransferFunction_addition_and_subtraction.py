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
def test_TransferFunction_addition_and_subtraction():
    tf1 = TransferFunction(s + 6, s - 5, s)
    tf2 = TransferFunction(s + 3, s + 1, s)
    tf3 = TransferFunction(s + 1, s ** 2 + s + 1, s)
    tf4 = TransferFunction(p, 2 - p, p)
    assert tf1 + tf2 == Parallel(tf1, tf2)
    assert tf3 + tf1 == Parallel(tf3, tf1)
    assert -tf1 + tf2 + tf3 == Parallel(-tf1, tf2, tf3)
    assert tf1 + (tf2 + tf3) == Parallel(tf1, tf2, tf3)
    c = symbols('c', commutative=False)
    raises(ValueError, lambda: tf1 + Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf2 + c)
    raises(ValueError, lambda: tf3 + tf4)
    raises(ValueError, lambda: tf1 + (s - 1))
    raises(ValueError, lambda: tf1 + 8)
    raises(ValueError, lambda: 1 - p ** 3 + tf1)
    assert tf1 - tf2 == Parallel(tf1, -tf2)
    assert tf3 - tf2 == Parallel(tf3, -tf2)
    assert -tf1 - tf3 == Parallel(-tf1, -tf3)
    assert tf1 - tf2 + tf3 == Parallel(tf1, -tf2, tf3)
    raises(ValueError, lambda: tf1 - Matrix([1, 2, 3]))
    raises(ValueError, lambda: tf3 - tf4)
    raises(ValueError, lambda: tf1 - (s - 1))
    raises(ValueError, lambda: tf1 - 8)
    raises(ValueError, lambda: s + 5 - tf2)
    raises(ValueError, lambda: 1 + p ** 4 - tf1)