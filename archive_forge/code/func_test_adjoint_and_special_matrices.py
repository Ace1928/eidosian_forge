from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_adjoint_and_special_matrices():
    A = Identity(3)
    B = OneMatrix(3, 2)
    C = ZeroMatrix(2, 3)
    D = Identity(2)
    X = BlockMatrix([[A, B], [C, D]])
    X2 = BlockMatrix([[A, S.ImaginaryUnit * B], [C, D]])
    assert X.adjoint() == BlockMatrix([[A, ZeroMatrix(3, 2)], [OneMatrix(2, 3), D]])
    assert re(X) == X
    assert X2.adjoint() == BlockMatrix([[A, ZeroMatrix(3, 2)], [-S.ImaginaryUnit * OneMatrix(2, 3), D]])
    assert im(X2) == BlockMatrix([[ZeroMatrix(3, 3), OneMatrix(3, 2)], [ZeroMatrix(2, 3), ZeroMatrix(2, 2)]])