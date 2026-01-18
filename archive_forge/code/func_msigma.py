from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated
def msigma(i):
    """Returns a Pauli matrix `\\sigma_i` with `i=1,2,3`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pauli_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import msigma
    >>> msigma(1)
    Matrix([
    [0, 1],
    [1, 0]])
    """
    if i == 1:
        mat = ((0, 1), (1, 0))
    elif i == 2:
        mat = ((0, -I), (I, 0))
    elif i == 3:
        mat = ((1, 0), (0, -1))
    else:
        raise IndexError('Invalid Pauli index')
    return Matrix(mat)