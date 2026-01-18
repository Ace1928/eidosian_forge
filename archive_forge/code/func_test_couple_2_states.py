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
def test_couple_2_states():
    assert JzKetCoupled(0, 0, (S.Half, S.Half)) == expand(couple(uncouple(JzKetCoupled(0, 0, (S.Half, S.Half)))))
    assert JzKetCoupled(1, 1, (S.Half, S.Half)) == expand(couple(uncouple(JzKetCoupled(1, 1, (S.Half, S.Half)))))
    assert JzKetCoupled(1, 0, (S.Half, S.Half)) == expand(couple(uncouple(JzKetCoupled(1, 0, (S.Half, S.Half)))))
    assert JzKetCoupled(1, -1, (S.Half, S.Half)) == expand(couple(uncouple(JzKetCoupled(1, -1, (S.Half, S.Half)))))
    assert JzKetCoupled(S.Half, S.Half, (1, S.Half)) == expand(couple(uncouple(JzKetCoupled(S.Half, S.Half, (1, S.Half)))))
    assert JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half)) == expand(couple(uncouple(JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half)))))
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half)) == expand(couple(uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half)))))
    assert JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half)) == expand(couple(uncouple(JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half)))))
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half)) == expand(couple(uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half)))))
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)) == expand(couple(uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)))))
    assert JzKetCoupled(0, 0, (1, 1)) == expand(couple(uncouple(JzKetCoupled(0, 0, (1, 1)))))
    assert JzKetCoupled(1, 1, (1, 1)) == expand(couple(uncouple(JzKetCoupled(1, 1, (1, 1)))))
    assert JzKetCoupled(1, 0, (1, 1)) == expand(couple(uncouple(JzKetCoupled(1, 0, (1, 1)))))
    assert JzKetCoupled(1, -1, (1, 1)) == expand(couple(uncouple(JzKetCoupled(1, -1, (1, 1)))))
    assert JzKetCoupled(2, 2, (1, 1)) == expand(couple(uncouple(JzKetCoupled(2, 2, (1, 1)))))
    assert JzKetCoupled(2, 1, (1, 1)) == expand(couple(uncouple(JzKetCoupled(2, 1, (1, 1)))))
    assert JzKetCoupled(2, 0, (1, 1)) == expand(couple(uncouple(JzKetCoupled(2, 0, (1, 1)))))
    assert JzKetCoupled(2, -1, (1, 1)) == expand(couple(uncouple(JzKetCoupled(2, -1, (1, 1)))))
    assert JzKetCoupled(2, -2, (1, 1)) == expand(couple(uncouple(JzKetCoupled(2, -2, (1, 1)))))
    assert JzKetCoupled(1, 1, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(1, 1, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(1, 0, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(1, 0, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(1, -1, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(1, -1, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(2, 2, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(2, 2, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(2, 1, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(2, 1, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(2, 0, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(2, 0, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(2, -1, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(2, -1, (S.Half, Rational(3, 2))))))
    assert JzKetCoupled(2, -2, (S.Half, Rational(3, 2))) == expand(couple(uncouple(JzKetCoupled(2, -2, (S.Half, Rational(3, 2))))))