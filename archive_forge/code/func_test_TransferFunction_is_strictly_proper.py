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
def test_TransferFunction_is_strictly_proper():
    omega_o, zeta, tau = symbols('omega_o, zeta, tau')
    tf1 = TransferFunction(omega_o ** 2, s ** 2 + p * omega_o * zeta * s + omega_o ** 2, omega_o)
    tf2 = TransferFunction(tau - s ** 3, tau + p ** 4, tau)
    tf3 = TransferFunction(a * b * s ** 3 + s ** 2 - a * p + s, b - s * p ** 2, p)
    tf4 = TransferFunction(b * s ** 2 + p ** 2 - a * p + s, b - p ** 2, s)
    assert not tf1.is_strictly_proper
    assert not tf2.is_strictly_proper
    assert tf3.is_strictly_proper
    assert not tf4.is_strictly_proper