from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def isprime(n):
    """
    Test if n is a prime number (True) or not (False). For n < 2^64 the
    answer is definitive; larger n values have a small probability of actually
    being pseudoprimes.

    Negative numbers (e.g. -2) are not considered prime.

    The first step is looking for trivial factors, which if found enables
    a quick return.  Next, if the sieve is large enough, use bisection search
    on the sieve.  For small numbers, a set of deterministic Miller-Rabin
    tests are performed with bases that are known to have no counterexamples
    in their range.  Finally if the number is larger than 2^64, a strong
    BPSW test is performed.  While this is a probable prime test and we
    believe counterexamples exist, there are no known counterexamples.

    Examples
    ========

    >>> from sympy.ntheory import isprime
    >>> isprime(13)
    True
    >>> isprime(13.0)  # limited precision
    False
    >>> isprime(15)
    False

    Notes
    =====

    This routine is intended only for integer input, not numerical
    expressions which may represent numbers. Floats are also
    rejected as input because they represent numbers of limited
    precision. While it is tempting to permit 7.0 to represent an
    integer there are errors that may "pass silently" if this is
    allowed:

    >>> from sympy import Float, S
    >>> int(1e3) == 1e3 == 10**3
    True
    >>> int(1e23) == 1e23
    True
    >>> int(1e23) == 10**23
    False

    >>> near_int = 1 + S(1)/10**19
    >>> near_int == int(near_int)
    False
    >>> n = Float(near_int, 10)  # truncated by precision
    >>> n == int(n)
    True
    >>> n = Float(near_int, 20)
    >>> n == int(n)
    False

    See Also
    ========

    sympy.ntheory.generate.primerange : Generates all primes in a given range
    sympy.ntheory.generate.primepi : Return the number of primes less than or equal to n
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========
    - https://en.wikipedia.org/wiki/Strong_pseudoprime
    - "Lucas Pseudoprimes", Baillie and Wagstaff, 1980.
      http://mpqs.free.fr/LucasPseudoprimes.pdf
    - https://en.wikipedia.org/wiki/Baillie-PSW_primality_test
    """
    try:
        n = as_int(n)
    except ValueError:
        return False
    if n in [2, 3, 5]:
        return True
    if n < 2 or n % 2 == 0 or n % 3 == 0 or (n % 5 == 0):
        return False
    if n < 49:
        return True
    if n % 7 == 0 or n % 11 == 0 or n % 13 == 0 or (n % 17 == 0) or (n % 19 == 0) or (n % 23 == 0) or (n % 29 == 0) or (n % 31 == 0) or (n % 37 == 0) or (n % 41 == 0) or (n % 43 == 0) or (n % 47 == 0):
        return False
    if n < 2809:
        return True
    if n < 31417:
        return pow(2, n, n) == 2 and n not in [7957, 8321, 13747, 18721, 19951, 23377]
    from sympy.ntheory.generate import sieve as s
    if n <= s._list[-1]:
        l, u = s.search(n)
        return l == u
    if HAS_GMPY == 2:
        from gmpy2 import is_strong_prp, is_strong_selfridge_prp
        return is_strong_prp(n, 2) and is_strong_selfridge_prp(n)
    if n < 341531:
        return mr(n, [9345883071009581737])
    if n < 885594169:
        return mr(n, [725270293939359937, 3569819667048198375])
    if n < 350269456337:
        return mr(n, [4230279247111683200, 14694767155120705706, 16641139526367750375])
    if n < 55245642489451:
        return mr(n, [2, 141889084524735, 1199124725622454117, 11096072698276303650])
    if n < 7999252175582851:
        return mr(n, [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805])
    if n < 585226005592931977:
        return mr(n, [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375])
    if n < 18446744073709551616:
        return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])
    if n < 318665857834031151167461:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    if n < 3317044064679887385961981:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    return mr(n, [2]) and is_strong_lucas_prp(n)