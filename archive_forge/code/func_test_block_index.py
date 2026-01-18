from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises
def test_block_index():
    I = Identity(3)
    Z = ZeroMatrix(3, 3)
    B = BlockMatrix([[I, I], [I, I]])
    e3 = ImmutableMatrix(eye(3))
    BB = BlockMatrix([[e3, e3], [e3, e3]])
    assert B[0, 0] == B[3, 0] == B[0, 3] == B[3, 3] == 1
    assert B[4, 3] == B[5, 1] == 0
    BB = BlockMatrix([[e3, e3], [e3, e3]])
    assert B.as_explicit() == BB.as_explicit()
    BI = BlockMatrix([[I, Z], [Z, I]])
    assert BI.as_explicit().equals(eye(6))