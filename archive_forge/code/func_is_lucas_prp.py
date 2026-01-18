from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def is_lucas_prp(n):
    """Standard Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a Lucas probable
    prime.

    This is typically used in combination with the Miller-Rabin test.

    References
    ==========
    - "Lucas Pseudoprimes", Baillie and Wagstaff, 1980.
      http://mpqs.free.fr/LucasPseudoprimes.pdf
    - OEIS A217120: Lucas Pseudoprimes
      https://oeis.org/A217120
    - https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_lucas_prp
    >>> for i in range(10000):
    ...     if is_lucas_prp(i) and not isprime(i):
    ...        print(i)
    323
    377
    1159
    1829
    3827
    5459
    5777
    9071
    9179
    """
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if is_square(n, False):
        return False
    D, P, Q = _lucas_selfridge_params(n)
    if D == 0:
        return False
    U, V, Qk = _lucas_sequence(n, P, Q, n + 1)
    return U == 0