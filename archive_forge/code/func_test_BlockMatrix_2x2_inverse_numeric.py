from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_BlockMatrix_2x2_inverse_numeric():
    """Test 2x2 block matrix inversion numerically for all 4 formulas"""
    M = Matrix([[1, 2], [3, 4]])
    D1 = Matrix([[1, 2], [2, 4]])
    D2 = Matrix([[1, 3], [3, 9]])
    D3 = Matrix([[1, 4], [4, 16]])
    assert D1.rank() == D2.rank() == D3.rank() == 1
    assert (D1 + D2).rank() == (D2 + D3).rank() == (D3 + D1).rank() == 2
    K = BlockMatrix([[M, D1], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    K = BlockMatrix([[D1, M], [D2, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    K = BlockMatrix([[D1, D2], [M, D3]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()
    K = BlockMatrix([[D1, D2], [D3, M]])
    assert block_collapse(K.inv()).as_explicit() == K.as_explicit().inv()