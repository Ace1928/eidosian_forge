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
def test_matrixelement_diff():
    dexpr = diff((D * w)[k, 0], w[p, 0])
    assert w[k, p].diff(w[k, p]) == 1
    assert w[k, p].diff(w[0, 0]) == KroneckerDelta(0, k, (0, n - 1)) * KroneckerDelta(0, p, (0, 0))
    _i_1 = Dummy('_i_1')
    assert dexpr.dummy_eq(Sum(KroneckerDelta(_i_1, p, (0, n - 1)) * D[k, _i_1], (_i_1, 0, n - 1)))
    assert dexpr.doit() == D[k, p]