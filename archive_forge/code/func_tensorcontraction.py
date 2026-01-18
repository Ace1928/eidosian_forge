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
def tensorcontraction(array, *contraction_axes):
    """
    Contraction of an array-like object on the specified axes.

    The equivalent operator for array expressions is ``ArrayContraction``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy import Array, tensorcontraction
    >>> from sympy import Matrix, eye
    >>> tensorcontraction(eye(3), (0, 1))
    3
    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensorcontraction(A, (0, 2))
    [21, 30]

    Matrix multiplication may be emulated with a proper combination of
    ``tensorcontraction`` and ``tensorproduct``

    >>> from sympy import tensorproduct
    >>> from sympy.abc import a,b,c,d,e,f,g,h
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> m2 = Matrix([[e, f], [g, h]])
    >>> p = tensorproduct(m1, m2)
    >>> p
    [[[[a*e, a*f], [a*g, a*h]], [[b*e, b*f], [b*g, b*h]]], [[[c*e, c*f], [c*g, c*h]], [[d*e, d*f], [d*g, d*h]]]]
    >>> tensorcontraction(p, (1, 2))
    [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    >>> m1*m2
    Matrix([
    [a*e + b*g, a*f + b*h],
    [c*e + d*g, c*f + d*h]])

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayContraction

    """
    from sympy.tensor.array.expressions.array_expressions import _array_contraction
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_contraction(array, *contraction_axes)
    array, remaining_indices, remaining_shape, summed_deltas = _util_contraction_diagonal(array, *contraction_axes)
    contracted_array = []
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = S.Zero
        for sum_to_index in itertools.product(*summed_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum += array[idx]
        contracted_array.append(isum)
    if len(remaining_indices) == 0:
        assert len(contracted_array) == 1
        return contracted_array[0]
    return type(array)(contracted_array, remaining_shape)