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
def test_innerproducts_of_rewritten_states():
    assert qapply(JxBra(1, 1) * JxKet(1, 1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 0) * JxKet(1, 0).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, -1) * JxKet(1, -1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 1) * JxKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JxBra(1, 0) * JxKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JxBra(1, -1) * JxKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 1) * JyKet(1, 1).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, 0) * JyKet(1, 0).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, -1) * JyKet(1, -1).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, 1) * JyKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 0) * JyKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, -1) * JyKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 1) * JyKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 0) * JyKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, -1) * JyKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JzBra(1, 1) * JzKet(1, 1).rewrite('Jy')).doit() == 1
    assert qapply(JzBra(1, 0) * JzKet(1, 0).rewrite('Jy')).doit() == 1
    assert qapply(JzBra(1, -1) * JzKet(1, -1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 1) * JxKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JxBra(1, 1) * JxKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 1) * JxKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JxBra(1, 1) * JxKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 1) * JyKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JyBra(1, 1) * JyKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 1) * JyKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JyBra(1, 1) * JyKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JzBra(1, 1) * JzKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JzBra(1, 1) * JzKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 1) * JzKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JzBra(1, 1) * JzKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0) * JxKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0) * JxKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0) * JxKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JxBra(1, 0) * JxKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 0) * JyKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 0) * JyKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 0) * JyKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 0) * JyKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JzBra(1, 0) * JzKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 0) * JzKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 0) * JzKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JzBra(1, 0) * JzKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, -1) * JxKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, -1) * JxKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JxBra(1, -1) * JxKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JxBra(1, -1) * JxKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JyBra(1, -1) * JyKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, -1) * JyKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JyBra(1, -1) * JyKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, -1) * JyKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JzBra(1, -1) * JzKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, -1) * JzKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JzBra(1, -1) * JzKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JzBra(1, -1) * JzKet(1, 0).rewrite('Jy')).doit() == 0