from sympy.core import S, sympify
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def covering_product(a, b):
    """
    Returns the covering product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The covering product of given sequences is a sequence which contains
    the sum of products of the elements of the given sequences grouped by
    the *bitwise-OR* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which covering product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, covering_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> covering_product([u, v], [x, y])
    [u*x, u*y + v*x + v*y]
    >>> covering_product([u, v, x], [y, z])
    [u*y, u*z + v*y + v*z, x*y, x*z]

    >>> covering_product([1, S(2)/3], [3, 4 + 5*I])
    [3, 26/3 + 25*I/3]
    >>> covering_product([1, 3, S(5)/7], [7, 8])
    [7, 53, 5, 40/7]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """
    if not a or not b:
        return []
    a, b = (a[:], b[:])
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    a, b = (mobius_transform(a), mobius_transform(b))
    a = [expand_mul(x * y) for x, y in zip(a, b)]
    a = inverse_mobius_transform(a)
    return a