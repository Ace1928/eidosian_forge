from sympy.core import S, sympify
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def convolution_fwht(a, b):
    """
    Performs dyadic (*bitwise-XOR*) convolution using Fast Walsh Hadamard
    Transform.

    The convolution is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S, I
    >>> from sympy.discrete.convolutions import convolution_fwht

    >>> u, v, x, y = symbols('u v x y')
    >>> convolution_fwht([u, v], [x, y])
    [u*x + v*y, u*y + v*x]

    >>> convolution_fwht([2, 3], [4, 5])
    [23, 22]
    >>> convolution_fwht([2, 5 + 4*I, 7], [6*I, 7, 3 + 4*I])
    [56 + 68*I, -10 + 30*I, 6 + 50*I, 48 + 32*I]

    >>> convolution_fwht([S(33)/7, S(55)/6, S(7)/4], [S(2)/3, 5])
    [2057/42, 1870/63, 7/6, 35/4]

    References
    ==========

    .. [1] https://www.radioeng.cz/fulltexts/2002/02_03_40_42.pdf
    .. [2] https://en.wikipedia.org/wiki/Hadamard_transform

    """
    if not a or not b:
        return []
    a, b = (a[:], b[:])
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    a, b = (fwht(a), fwht(b))
    a = [expand_mul(x * y) for x, y in zip(a, b)]
    a = ifwht(a)
    return a