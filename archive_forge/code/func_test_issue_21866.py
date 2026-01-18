from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_issue_21866():
    n = 10
    I = Identity(n)
    O = ZeroMatrix(n, n)
    A = BlockMatrix([[I, O, O, O], [O, I, O, O], [O, O, I, O], [I, O, O, I]])
    Ainv = block_collapse(A.inv())
    AinvT = BlockMatrix([[I, O, O, O], [O, I, O, O], [O, O, I, O], [-I, O, O, I]])
    assert Ainv == AinvT