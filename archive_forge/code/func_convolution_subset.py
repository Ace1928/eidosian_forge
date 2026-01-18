from sympy.core import S, sympify
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def convolution_subset(a, b):
    """
    Performs Subset Convolution of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.discrete.convolutions import convolution_subset
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> convolution_subset([u, v], [x, y])
    [u*x, u*y + v*x]
    >>> convolution_subset([u, v, x], [y, z])
    [u*y, u*z + v*y, x*y, x*z]

    >>> convolution_subset([1, S(2)/3], [3, 4])
    [3, 6]
    >>> convolution_subset([1, 3, S(5)/7], [7])
    [7, 21, 5, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """
    if not a or not b:
        return []
    if not iterable(a) or not iterable(b):
        raise TypeError('Expected a sequence of coefficients for convolution')
    a = [sympify(arg) for arg in a]
    b = [sympify(arg) for arg in b]
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    c = [S.Zero] * n
    for mask in range(n):
        smask = mask
        while smask > 0:
            c[mask] += expand_mul(a[smask] * b[mask ^ smask])
            smask = smask - 1 & mask
        c[mask] += expand_mul(a[smask] * b[mask ^ smask])
    return c