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
def tensorproduct(*args):
    """
    Tensor product among scalars or array-like objects.

    The equivalent operator for array expressions is ``ArrayTensorProduct``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.tensor.array import tensorproduct, Array
    >>> from sympy.abc import x, y, z, t
    >>> A = Array([[1, 2], [3, 4]])
    >>> B = Array([x, y])
    >>> tensorproduct(A, B)
    [[[x, y], [2*x, 2*y]], [[3*x, 3*y], [4*x, 4*y]]]
    >>> tensorproduct(A, x)
    [[x, 2*x], [3*x, 4*x]]
    >>> tensorproduct(A, B, B)
    [[[[x**2, x*y], [x*y, y**2]], [[2*x**2, 2*x*y], [2*x*y, 2*y**2]]], [[[3*x**2, 3*x*y], [3*x*y, 3*y**2]], [[4*x**2, 4*x*y], [4*x*y, 4*y**2]]]]

    Applying this function on two matrices will result in a rank 4 array.

    >>> from sympy import Matrix, eye
    >>> m = Matrix([[x, y], [z, t]])
    >>> p = tensorproduct(eye(3), m)
    >>> p
    [[[[x, y], [z, t]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[x, y], [z, t]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[x, y], [z, t]]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayTensorProduct

    """
    from sympy.tensor.array import SparseNDimArray, ImmutableSparseNDimArray
    if len(args) == 0:
        return S.One
    if len(args) == 1:
        return _arrayfy(args[0])
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if any((isinstance(arg, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)) for arg in args)):
        return ArrayTensorProduct(*args)
    if len(args) > 2:
        return tensorproduct(tensorproduct(args[0], args[1]), *args[2:])
    a, b = map(_arrayfy, args)
    if not isinstance(a, NDimArray) or not isinstance(b, NDimArray):
        return a * b
    if isinstance(a, SparseNDimArray) and isinstance(b, SparseNDimArray):
        lp = len(b)
        new_array = {k1 * lp + k2: v1 * v2 for k1, v1 in a._sparse_array.items() for k2, v2 in b._sparse_array.items()}
        return ImmutableSparseNDimArray(new_array, a.shape + b.shape)
    product_list = [i * j for i in Flatten(a) for j in Flatten(b)]
    return ImmutableDenseNDimArray(product_list, a.shape + b.shape)