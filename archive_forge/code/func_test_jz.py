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
def test_jz():
    assert Commutator(Jz, Jminus).doit() == -hbar * Jminus
    assert qapply(Jz * JxKet(1, 1)) == -sqrt(2) * hbar * JxKet(1, 0) / 2
    assert qapply(Jz * JyKet(1, 1)) == -sqrt(2) * hbar * I * JyKet(1, 0) / 2
    assert qapply(Jz * JzKet(2, 1)) == hbar * JzKet(2, 1)
    assert qapply(Jz * JxKet(j, m)) == Sum(hbar * mi * WignerD(j, mi, m, 0, pi / 2, 0) * Sum(WignerD(j, mi1, mi, 0, pi * Rational(3, 2), 0) * JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz * JyKet(j, m)) == Sum(hbar * mi * WignerD(j, mi, m, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j, mi1, mi, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz * JzKet(j, m)) == hbar * m * JzKet(j, m)
    assert qapply(Jz * JxKetCoupled(1, 1, (1, 1))) == -sqrt(2) * hbar * JxKetCoupled(1, 0, (1, 1)) / 2
    assert qapply(Jz * JyKetCoupled(1, 1, (1, 1))) == -sqrt(2) * hbar * I * JyKetCoupled(1, 0, (1, 1)) / 2
    assert qapply(Jz * JzKetCoupled(1, 1, (1, 1))) == hbar * JzKetCoupled(1, 1, (1, 1))
    assert qapply(Jz * JxKetCoupled(j, m, (j1, j2))) == Sum(hbar * mi * WignerD(j, mi, m, 0, pi / 2, 0) * Sum(WignerD(j, mi1, mi, 0, pi * Rational(3, 2), 0) * JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz * JyKetCoupled(j, m, (j1, j2))) == Sum(hbar * mi * WignerD(j, mi, m, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j, mi1, mi, pi * Rational(3, 2), pi / 2, pi / 2) * JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz * JzKetCoupled(j, m, (j1, j2))) == hbar * m * JzKetCoupled(j, m, (j1, j2))
    assert qapply(Jz * TensorProduct(JxKet(1, 1), JxKet(1, 1))) == -sqrt(2) * hbar * TensorProduct(JxKet(1, 1), JxKet(1, 0)) / 2 - sqrt(2) * hbar * TensorProduct(JxKet(1, 0), JxKet(1, 1)) / 2
    assert qapply(Jz * TensorProduct(JyKet(1, 1), JyKet(1, 1))) == -sqrt(2) * hbar * I * TensorProduct(JyKet(1, 1), JyKet(1, 0)) / 2 - sqrt(2) * hbar * I * TensorProduct(JyKet(1, 0), JyKet(1, 1)) / 2
    assert qapply(Jz * TensorProduct(JzKet(1, 1), JzKet(1, 1))) == 2 * hbar * TensorProduct(JzKet(1, 1), JzKet(1, 1))
    assert qapply(Jz * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0
    assert qapply(Jz * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(JxKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, 0, pi / 2, 0) * Sum(WignerD(j2, mi1, mi, 0, pi * Rational(3, 2), 0) * JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, 0, pi / 2, 0) * Sum(WignerD(j1, mi1, mi, 0, pi * Rational(3, 2), 0) * JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(Jz * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(JyKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j2, mi1, mi, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j1, mi1, mi, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(Jz * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * m1 * TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) + hbar * m2 * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    assert qapply(TensorProduct(Jz, 1) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == -sqrt(2) * hbar * TensorProduct(JxKet(1, 0), JxKet(1, -1)) / 2
    assert qapply(TensorProduct(1, Jz) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == -sqrt(2) * hbar * TensorProduct(JxKet(1, 1), JxKet(1, 0)) / 2
    assert qapply(TensorProduct(Jz, 1) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == -sqrt(2) * I * hbar * TensorProduct(JyKet(1, 0), JyKet(1, -1)) / 2
    assert qapply(TensorProduct(1, Jz) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == sqrt(2) * I * hbar * TensorProduct(JyKet(1, 1), JyKet(1, 0)) / 2
    assert qapply(TensorProduct(Jz, 1) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == hbar * TensorProduct(JzKet(1, 1), JzKet(1, -1))
    assert qapply(TensorProduct(1, Jz) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == -hbar * TensorProduct(JzKet(1, 1), JzKet(1, -1))
    assert qapply(TensorProduct(Jz, 1) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, 0, pi / 2, 0) * Sum(WignerD(j1, mi1, mi, 0, pi * Rational(3, 2), 0) * JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jz) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(JxKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, 0, pi / 2, 0) * Sum(WignerD(j2, mi1, mi, 0, pi * Rational(3, 2), 0) * JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jz, 1) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j1, mi1, mi, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jz) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(JyKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j2, mi1, mi, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jz, 1) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * m1 * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    assert qapply(TensorProduct(1, Jz) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * m2 * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))