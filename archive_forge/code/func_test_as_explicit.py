from sympy.core.expr import unchanged
from sympy.core.symbol import Symbol, symbols
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.companion import CompanionMatrix
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
def test_as_explicit():
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    assert CompanionMatrix(Poly([1, c0], x)).as_explicit() == ImmutableDenseMatrix([-c0])
    assert CompanionMatrix(Poly([1, c1, c0], x)).as_explicit() == ImmutableDenseMatrix([[0, -c0], [1, -c1]])
    assert CompanionMatrix(Poly([1, c2, c1, c0], x)).as_explicit() == ImmutableDenseMatrix([[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]])