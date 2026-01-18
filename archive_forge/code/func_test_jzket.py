from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.abc import alpha, beta, gamma, j, m
from sympy.physics.quantum import hbar, represent, Commutator, InnerProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import (
from sympy.testing.pytest import raises, slow
def test_jzket():
    j, m = symbols('j m')
    raises(ValueError, lambda: JzKet(Rational(2, 3), Rational(-1, 3)))
    raises(ValueError, lambda: JzKet(Rational(2, 3), m))
    raises(ValueError, lambda: JzKet(-1, 1))
    raises(ValueError, lambda: JzKet(-1, m))
    raises(ValueError, lambda: JzKet(j, Rational(-1, 3)))
    raises(ValueError, lambda: JzKet(1, 2))
    raises(ValueError, lambda: JzKet(1, -2))
    raises(ValueError, lambda: JzKet(1, S.Half))