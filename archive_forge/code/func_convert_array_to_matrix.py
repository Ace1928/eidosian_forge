import itertools
from collections import defaultdict
from typing import Tuple as tTuple, Union as tUnion, FrozenSet, Dict as tDict, List, Optional
from functools import singledispatch
from itertools import accumulate
from sympy import MatMul, Basic, Wild, KroneckerProduct
from sympy.assumptions.ask import (Q, ask)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert, Permutation
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks
def convert_array_to_matrix(expr):
    """
    Recognize matrix expressions in codegen objects.

    If more than one matrix multiplication line have been detected, return a
    list with the matrix expressions.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy.tensor.array import tensorcontraction, tensorproduct
    >>> from sympy import MatrixSymbol, Sum
    >>> from sympy.abc import i, j, k, l, N
    >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> A = MatrixSymbol("A", N, N)
    >>> B = MatrixSymbol("B", N, N)
    >>> C = MatrixSymbol("C", N, N)
    >>> D = MatrixSymbol("D", N, N)

    >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A.T

    Transposition is detected:

    >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A.T*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A

    Detect the trace:

    >>> expr = Sum(A[i, i], (i, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A)

    Recognize some more complex traces:

    >>> expr = Sum(A[i, j]*B[j, i], (i, 0, N-1), (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A*B)

    More complicated expressions:

    >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B.T*A.T

    Expressions constructed from matrix expressions do not contain literal
    indices, the positions of free indices are returned instead:

    >>> expr = A*B
    >>> cg = convert_matrix_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B

    If more than one line of matrix multiplications is detected, return
    separate matrix multiplication factors embedded in a tensor product object:

    >>> cg = tensorcontraction(tensorproduct(A, B, C, D), (1, 2), (5, 6))
    >>> convert_array_to_matrix(cg)
    ArrayTensorProduct(A*B, C*D)

    The two lines have free indices at axes 0, 3 and 4, 7, respectively.
    """
    rec = _array2matrix(expr)
    rec, removed = _remove_trivial_dims(rec)
    return rec