from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_issue_17624():
    a = MatrixSymbol('a', 2, 2)
    z = ZeroMatrix(2, 2)
    b = BlockMatrix([[a, z], [z, z]])
    assert block_collapse(b * b) == BlockMatrix([[a ** 2, z], [z, z]])
    assert block_collapse(b * b * b) == BlockMatrix([[a ** 3, z], [z, z]])