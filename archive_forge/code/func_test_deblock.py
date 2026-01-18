from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_deblock():
    B = BlockMatrix([[MatrixSymbol('A_%d%d' % (i, j), n, n) for j in range(4)] for i in range(4)])
    assert deblock(reblock_2x2(B)) == B