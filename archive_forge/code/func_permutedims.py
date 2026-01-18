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
def permutedims(expr, perm=None, index_order_old=None, index_order_new=None):
    """
    Permutes the indices of an array.

    Parameter specifies the permutation of the indices.

    The equivalent operator for array expressions is ``PermuteDims``, which can
    be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.abc import x, y, z, t
    >>> from sympy import sin
    >>> from sympy import Array, permutedims
    >>> a = Array([[x, y, z], [t, sin(x), 0]])
    >>> a
    [[x, y, z], [t, sin(x), 0]]
    >>> permutedims(a, (1, 0))
    [[x, t], [y, sin(x)], [z, 0]]

    If the array is of second order, ``transpose`` can be used:

    >>> from sympy import transpose
    >>> transpose(a)
    [[x, t], [y, sin(x)], [z, 0]]

    Examples on higher dimensions:

    >>> b = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> permutedims(b, (2, 1, 0))
    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    >>> permutedims(b, (1, 2, 0))
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    An alternative way to specify the same permutations as in the previous
    lines involves passing the *old* and *new* indices, either as a list or as
    a string:

    >>> permutedims(b, index_order_old="cba", index_order_new="abc")
    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    >>> permutedims(b, index_order_old="cab", index_order_new="abc")
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    ``Permutation`` objects are also allowed:

    >>> from sympy.combinatorics import Permutation
    >>> permutedims(b, Permutation([1, 2, 0]))
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.PermuteDims

    """
    from sympy.tensor.array import SparseNDimArray
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _permute_dims
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.tensor.array.expressions import PermuteDims
    from sympy.tensor.array.expressions.array_expressions import get_rank
    perm = PermuteDims._get_permutation_from_arguments(perm, index_order_old, index_order_new, get_rank(expr))
    if isinstance(expr, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _permute_dims(expr, perm)
    if not isinstance(expr, NDimArray):
        expr = ImmutableDenseNDimArray(expr)
    from sympy.combinatorics import Permutation
    if not isinstance(perm, Permutation):
        perm = Permutation(list(perm))
    if perm.size != expr.rank():
        raise ValueError('wrong permutation size')
    iperm = ~perm
    new_shape = perm(expr.shape)
    if isinstance(expr, SparseNDimArray):
        return type(expr)({tuple(perm(expr._get_tuple_index(k))): v for k, v in expr._sparse_array.items()}, new_shape)
    indices_span = perm([range(i) for i in expr.shape])
    new_array = [None] * len(expr)
    for i, idx in enumerate(itertools.product(*indices_span)):
        t = iperm(idx)
        new_array[i] = expr[t]
    return type(expr)(new_array, new_shape)