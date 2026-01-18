from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_invalid_block_matrix():
    raises(ValueError, lambda: BlockMatrix([[Identity(2), Identity(5)]]))
    raises(ValueError, lambda: BlockMatrix([[Identity(n), Identity(m)]]))
    raises(ValueError, lambda: BlockMatrix([[ZeroMatrix(n, n), ZeroMatrix(n, n)], [ZeroMatrix(n, n - 1), ZeroMatrix(n, n + 1)]]))
    raises(ValueError, lambda: BlockMatrix([[ZeroMatrix(n - 1, n), ZeroMatrix(n, n)], [ZeroMatrix(n + 1, n), ZeroMatrix(n, n)]]))