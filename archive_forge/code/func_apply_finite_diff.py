from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
def apply_finite_diff(order, x_list, y_list, x0=S.Zero):
    """
    Calculates the finite difference approximation of
    the derivative of requested order at ``x0`` from points
    provided in ``x_list`` and ``y_list``.

    Parameters
    ==========

    order: int
        order of derivative to approximate. 0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
    y_list: sequence
        The function value at corresponding values for the independent
        variable in x_list.
    x0: Number or Symbol
        At what value of the independent variable the derivative should be
        evaluated. Defaults to 0.

    Returns
    =======

    sympy.core.add.Add or sympy.core.numbers.Number
        The finite difference expression approximating the requested
        derivative order at ``x0``.

    Examples
    ========

    >>> from sympy import apply_finite_diff
    >>> cube = lambda arg: (1.0*arg)**3
    >>> xlist = range(-3,3+1)
    >>> apply_finite_diff(2, xlist, map(cube, xlist), 2) - 12 # doctest: +SKIP
    -3.55271367880050e-15

    we see that the example above only contain rounding errors.
    apply_finite_diff can also be used on more abstract objects:

    >>> from sympy import IndexedBase, Idx
    >>> x, y = map(IndexedBase, 'xy')
    >>> i = Idx('i')
    >>> x_list, y_list = zip(*[(x[i+j], y[i+j]) for j in range(-1,2)])
    >>> apply_finite_diff(1, x_list, y_list, x[i])
    ((x[i + 1] - x[i])/(-x[i - 1] + x[i]) - 1)*y[i]/(x[i + 1] - x[i]) -
    (x[i + 1] - x[i])*y[i - 1]/((x[i + 1] - x[i - 1])*(-x[i - 1] + x[i])) +
    (-x[i - 1] + x[i])*y[i + 1]/((x[i + 1] - x[i - 1])*(x[i + 1] - x[i]))

    Notes
    =====

    Order = 0 corresponds to interpolation.
    Only supply so many points you think makes sense
    to around x0 when extracting the derivative (the function
    need to be well behaved within that region). Also beware
    of Runge's phenomenon.

    See also
    ========

    sympy.calculus.finite_diff.finite_diff_weights

    References
    ==========

    Fortran 90 implementation with Python interface for numerics: finitediff_

    .. _finitediff: https://github.com/bjodah/finitediff

    """
    N = len(x_list) - 1
    if len(x_list) != len(y_list):
        raise ValueError('x_list and y_list not equal in length.')
    delta = finite_diff_weights(order, x_list, x0)
    derivative = 0
    for nu in range(len(x_list)):
        derivative += delta[order][N][nu] * y_list[nu]
    return derivative