from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixReductions
from sympy.testing.pytest import raises
from sympy.matrices import Matrix, zeros
from sympy.core.symbol import Symbol
from sympy.core.numbers import Rational
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.abc import x
def eye_Reductions(n):
    return ReductionsOnlyMatrix(n, n, lambda i, j: int(i == j))