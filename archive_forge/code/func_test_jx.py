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
def test_jx():
    assert Commutator(Jx, Jz).doit() == -I * hbar * Jy
    assert Jx.rewrite('plusminus') == (Jminus + Jplus) / 2
    assert represent(Jx, basis=Jz, j=1) == (represent(Jplus, basis=Jz, j=1) + represent(Jminus, basis=Jz, j=1)) / 2
    assert qapply(Jx * JxKet(1, 1)) == hbar * JxKet(1, 1)
    assert qapply(Jx * JyKet(1, 1)) == hbar * JyKet(1, 1)
    assert qapply(Jx * JzKet(1, 1)) == sqrt(2) * hbar * JzKet(1, 0) / 2
    assert qapply(Jx * JxKet(j, m)) == hbar * m * JxKet(j, m)
    assert qapply(Jx * JyKet(j, m)) == Sum(hbar * mi * WignerD(j, mi, m, 0, 0, pi / 2) * Sum(WignerD(j, mi1, mi, pi * Rational(3, 2), 0, 0) * JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jx * JzKet(j, m)) == hbar * sqrt(j ** 2 + j - m ** 2 - m) * JzKet(j, m + 1) / 2 + hbar * sqrt(j ** 2 + j - m ** 2 + m) * JzKet(j, m - 1) / 2
    assert qapply(Jx * JxKetCoupled(1, 1, (1, 1))) == hbar * JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jx * JyKetCoupled(1, 1, (1, 1))) == hbar * JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jx * JzKetCoupled(1, 1, (1, 1))) == sqrt(2) * hbar * JzKetCoupled(1, 0, (1, 1)) / 2
    assert qapply(Jx * JxKetCoupled(j, m, (j1, j2))) == hbar * m * JxKetCoupled(j, m, (j1, j2))
    assert qapply(Jx * JyKetCoupled(j, m, (j1, j2))) == Sum(hbar * mi * WignerD(j, mi, m, 0, 0, pi / 2) * Sum(WignerD(j, mi1, mi, pi * Rational(3, 2), 0, 0) * JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jx * JzKetCoupled(j, m, (j1, j2))) == hbar * sqrt(j ** 2 + j - m ** 2 - m) * JzKetCoupled(j, m + 1, (j1, j2)) / 2 + hbar * sqrt(j ** 2 + j - m ** 2 + m) * JzKetCoupled(j, m - 1, (j1, j2)) / 2
    assert qapply(Jx * TensorProduct(JxKet(1, 1), JxKet(1, 1))) == 2 * hbar * TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert qapply(Jx * TensorProduct(JyKet(1, 1), JyKet(1, 1))) == hbar * TensorProduct(JyKet(1, 1), JyKet(1, 1)) + hbar * TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert qapply(Jx * TensorProduct(JzKet(1, 1), JzKet(1, 1))) == sqrt(2) * hbar * TensorProduct(JzKet(1, 1), JzKet(1, 0)) / 2 + sqrt(2) * hbar * TensorProduct(JzKet(1, 0), JzKet(1, 1)) / 2
    assert qapply(Jx * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == 0
    assert qapply(Jx * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == hbar * m1 * TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) + hbar * m2 * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(Jx * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, 0, 0, pi / 2) * Sum(WignerD(j1, mi1, mi, pi * Rational(3, 2), 0, 0) * JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2)) + TensorProduct(JyKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, 0, 0, pi / 2) * Sum(WignerD(j2, mi1, mi, pi * Rational(3, 2), 0, 0) * JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(Jx * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * sqrt(j1 ** 2 + j1 - m1 ** 2 - m1) * TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2)) / 2 + hbar * sqrt(j1 ** 2 + j1 - m1 ** 2 + m1) * TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2)) / 2 + hbar * sqrt(j2 ** 2 + j2 - m2 ** 2 - m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1)) / 2 + hbar * sqrt(j2 ** 2 + j2 - m2 ** 2 + m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1)) / 2
    assert qapply(TensorProduct(Jx, 1) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == hbar * TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jx) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == -hbar * TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(Jx, 1) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == hbar * TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jx) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == -hbar * TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(Jx, 1) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == hbar * sqrt(2) * TensorProduct(JzKet(1, 0), JzKet(1, -1)) / 2
    assert qapply(TensorProduct(1, Jx) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == hbar * sqrt(2) * TensorProduct(JzKet(1, 1), JzKet(1, 0)) / 2
    assert qapply(TensorProduct(Jx, 1) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == hbar * m1 * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jx) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == hbar * m2 * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))
    assert qapply(TensorProduct(Jx, 1) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, 0, 0, pi / 2) * Sum(WignerD(j1, mi1, mi, pi * Rational(3, 2), 0, 0) * JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jx) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(JyKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, 0, 0, pi / 2) * Sum(WignerD(j2, mi1, mi, pi * Rational(3, 2), 0, 0) * JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jx, 1) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * sqrt(j1 ** 2 + j1 - m1 ** 2 - m1) * TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2)) / 2 + hbar * sqrt(j1 ** 2 + j1 - m1 ** 2 + m1) * TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2)) / 2
    assert qapply(TensorProduct(1, Jx) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * sqrt(j2 ** 2 + j2 - m2 ** 2 - m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1)) / 2 + hbar * sqrt(j2 ** 2 + j2 - m2 ** 2 + m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1)) / 2