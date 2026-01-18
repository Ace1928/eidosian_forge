from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor
from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL
def test_matrix_symbol_creation():
    assert MatrixSymbol('A', 2, 2)
    assert MatrixSymbol('A', 0, 0)
    raises(ValueError, lambda: MatrixSymbol('A', -1, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2.0, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2j, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2, -1))
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2.0))
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2j))
    n = symbols('n')
    assert MatrixSymbol('A', n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))