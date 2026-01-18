from sympy.core import S, sympify
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def convolution_fft(a, b, dps=None):
    """
    Performs linear convolution using Fast Fourier Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import S, I
    >>> from sympy.discrete.convolutions import convolution_fft

    >>> convolution_fft([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_fft([2, 5], [6, 7, 3])
    [12, 44, 41, 15]
    >>> convolution_fft([1 + 2*I, 4 + 3*I], [S(5)/4, 6])
    [5/4 + 5*I/2, 11 + 63*I/4, 24 + 18*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """
    a, b = (a[:], b[:])
    n = m = len(a) + len(b) - 1
    if n > 0 and n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    a, b = (fft(a, dps), fft(b, dps))
    a = [expand_mul(x * y) for x, y in zip(a, b)]
    a = ifft(a, dps)[:m]
    return a