from sympy.core.add import Add
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.core.relational import Eq
from sympy.concrete.summations import Sum
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import (
from sympy.matrices.expressions.matmul import MatMul
from sympy.testing.pytest import raises
def test_identity_matrix_creation():
    assert Identity(2)
    assert Identity(0)
    raises(ValueError, lambda: Identity(-1))
    raises(ValueError, lambda: Identity(2.0))
    raises(ValueError, lambda: Identity(2j))
    n = symbols('n')
    assert Identity(n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: Identity(n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: Identity(n))