import itertools
from collections.abc import Iterable
from sympy.core._print_helpers import Printable
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray
def tensordiagonal(array, *diagonal_axes):
    """
    Diagonalization of an array-like object on the specified axes.

    This is equivalent to multiplying the expression by Kronecker deltas
    uniting the axes.

    The diagonal indices are put at the end of the axes.

    The equivalent operator for array expressions is ``ArrayDiagonal``, which
    can be used to keep the expression unevaluated.

    Examples
    ========

    ``tensordiagonal`` acting on a 2-dimensional array by axes 0 and 1 is
    equivalent to the diagonal of the matrix:

    >>> from sympy import Array, tensordiagonal
    >>> from sympy import Matrix, eye
    >>> tensordiagonal(eye(3), (0, 1))
    [1, 1, 1]

    >>> from sympy.abc import a,b,c,d
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> tensordiagonal(m1, [0, 1])
    [a, d]

    In case of higher dimensional arrays, the diagonalized out dimensions
    are appended removed and appended as a single dimension at the end:

    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensordiagonal(A, (0, 2))
    [[0, 7, 14], [3, 10, 17]]
    >>> from sympy import permutedims
    >>> tensordiagonal(A, (0, 2)) == permutedims(Array([A[0, :, 0], A[1, :, 1], A[2, :, 2]]), [1, 0])
    True

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayDiagonal

    """
    if any((len(i) <= 1 for i in diagonal_axes)):
        raise ValueError('need at least two axes to diagonalize')
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, _array_diagonal
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_diagonal(array, *diagonal_axes)
    ArrayDiagonal._validate(array, *diagonal_axes)
    array, remaining_indices, remaining_shape, diagonal_deltas = _util_contraction_diagonal(array, *diagonal_axes)
    diagonalized_array = []
    diagonal_shape = [len(i) for i in diagonal_deltas]
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = []
        for sum_to_index in itertools.product(*diagonal_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum.append(array[idx])
        isum = type(array)(isum).reshape(*diagonal_shape)
        diagonalized_array.append(isum)
    return type(array)(diagonalized_array, remaining_shape + diagonal_shape)