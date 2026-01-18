from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_block_collapse_explicit_matrices():
    A = Matrix([[1, 2], [3, 4]])
    assert block_collapse(BlockMatrix([[A]])) == A
    A = ImmutableSparseMatrix([[1, 2], [3, 4]])
    assert block_collapse(BlockMatrix([[A]])) == A