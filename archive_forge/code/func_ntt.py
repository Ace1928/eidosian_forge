from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int
def ntt(seq, prime):
    """
    Performs the Number Theoretic Transform (**NTT**), which specializes the
    Discrete Fourier Transform (**DFT**) over quotient ring `Z/pZ` for prime
    `p` instead of complex numbers `C`.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 NTT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy import ntt, intt
    >>> ntt([1, 2, 3, 4], prime=3*2**8 + 1)
    [10, 643, 767, 122]
    >>> intt(_, 3*2**8 + 1)
    [1, 2, 3, 4]
    >>> intt([1, 2, 3, 4], prime=3*2**8 + 1)
    [387, 415, 384, 353]
    >>> ntt(_, prime=3*2**8 + 1)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] http://www.apfloat.org/ntt.html
    .. [2] https://mathworld.wolfram.com/NumberTheoreticTransform.html
    .. [3] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """
    return _number_theoretic_transform(seq, prime=prime)