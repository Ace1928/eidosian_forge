from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
def partial_transpose(expr, dims: Tuple[int, ...], axis: Optional[int]=0):
    """
    Assumes :math:`\\texttt{expr} = X_1 \\otimes ... \\otimes X_n` is a 2D Kronecker
    product composed of :math:`n = \\texttt{len(dims)}` implicit subsystems.
    Letting :math:`k = \\texttt{axis}`, the returned expression is a
    *partial transpose* of :math:`\\texttt{expr}`, with the transpose applied to its
    :math:`k^{\\text{th}}` implicit subsystem:

    .. math::
        X_1 \\otimes ... \\otimes X_k^T \\otimes ... \\otimes X_n.

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The 2D expression to take the partial transpose of.
    dims : tuple of ints.
        A tuple of integers encoding the dimensions of each subsystem.
    axis : int
        The index of the subsystem to be transposed
        from the tensor product that defines expr.
    """
    expr = Atom.cast_to_const(expr)
    if expr.ndim < 2 or expr.shape[0] != expr.shape[1]:
        raise ValueError('Only supports square matrices.')
    if axis < 0 or axis >= len(dims):
        raise ValueError(f'Invalid axis argument, should be between 0 and {len(dims)}, got {axis}.')
    if expr.shape[0] != np.prod(dims):
        raise ValueError("Dimension of system doesn't correspond to dimension of subsystems.")
    return sum([_term(expr, i, j, dims, axis) for i in range(dims[axis]) for j in range(dims[axis])])