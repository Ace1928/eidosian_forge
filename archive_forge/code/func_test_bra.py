from sympy.core.add import Add
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.state import (
from sympy.physics.quantum.hilbert import HilbertSpace
def test_bra():
    b = Bra('0')
    assert isinstance(b, Bra)
    assert isinstance(b, BraBase)
    assert isinstance(b, StateBase)
    assert isinstance(b, QExpr)
    assert b.label == (Symbol('0'),)
    assert b.hilbert_space == HilbertSpace()
    assert b.is_commutative is False
    b = Bra('pi')
    assert b.label == (Symbol('pi'),)
    b = Bra(x, y)
    assert b.label == (x, y)
    assert b.hilbert_space == HilbertSpace()
    assert b.is_commutative is False
    assert b.dual_class() == Ket
    assert b.dual == Ket(x, y)
    assert b.subs(x, y) == Bra(y, y)
    assert Bra() == Bra('psi')