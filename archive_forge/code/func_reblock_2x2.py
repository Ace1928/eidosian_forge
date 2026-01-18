from sympy.assumptions.ask import (Q, ask)
from sympy.core import Basic, Add, Mul, S
from sympy.core.sympify import _sympify
from sympy.functions import adjoint
from sympy.functions.elementary.complexes import re, im
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent
from sympy.matrices import Matrix, ShapeError
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.matrices.expressions.trace import trace
from sympy.matrices.expressions.transpose import Transpose, transpose
def reblock_2x2(expr):
    """
    Reblock a BlockMatrix so that it has 2x2 blocks of block matrices.  If
    possible in such a way that the matrix continues to be invertible using the
    classical 2x2 block inversion formulas.
    """
    if not isinstance(expr, BlockMatrix) or not all((d > 2 for d in expr.blockshape)):
        return expr
    BM = BlockMatrix
    rowblocks, colblocks = expr.blockshape
    blocks = expr.blocks
    for i in range(1, rowblocks):
        for j in range(1, colblocks):
            A = bc_unpack(BM(blocks[:i, :j]))
            B = bc_unpack(BM(blocks[:i, j:]))
            C = bc_unpack(BM(blocks[i:, :j]))
            D = bc_unpack(BM(blocks[i:, j:]))
            formula = _choose_2x2_inversion_formula(A, B, C, D)
            if formula is not None:
                return BlockMatrix([[A, B], [C, D]])
    return BM([[blocks[0, 0], BM(blocks[0, 1:])], [BM(blocks[1:, 0]), BM(blocks[1:, 1:])]])