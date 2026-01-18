from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int
def fwht(seq):
    """
    Performs the Walsh Hadamard Transform (**WHT**), and uses Hadamard
    ordering for the sequence.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which WHT is to be applied.

    Examples
    ========

    >>> from sympy import fwht, ifwht
    >>> fwht([4, 2, 2, 0, 0, 2, -2, 0])
    [8, 0, 8, 0, 8, 8, 0, 0]
    >>> ifwht(_)
    [4, 2, 2, 0, 0, 2, -2, 0]

    >>> ifwht([19, -1, 11, -9, -7, 13, -15, 5])
    [2, 0, 4, 0, 3, 10, 0, 0]
    >>> fwht(_)
    [19, -1, 11, -9, -7, 13, -15, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_transform
    .. [2] https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

    """
    return _walsh_hadamard_transform(seq)