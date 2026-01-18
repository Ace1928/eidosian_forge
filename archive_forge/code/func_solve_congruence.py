from functools import reduce
from math import prod
from sympy.core.numbers import igcdex, igcd
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int
def solve_congruence(*remainder_modulus_pairs, **hint):
    """Compute the integer ``n`` that has the residual ``ai`` when it is
    divided by ``mi`` where the ``ai`` and ``mi`` are given as pairs to
    this function: ((a1, m1), (a2, m2), ...). If there is no solution,
    return None. Otherwise return ``n`` and its modulus.

    The ``mi`` values need not be co-prime. If it is known that the moduli are
    not co-prime then the hint ``check`` can be set to False (default=True) and
    the check for a quicker solution via crt() (valid when the moduli are
    co-prime) will be skipped.

    If the hint ``symmetric`` is True (default is False), the value of ``n``
    will be within 1/2 of the modulus, possibly negative.

    Examples
    ========

    >>> from sympy.ntheory.modular import solve_congruence

    What number is 2 mod 3, 3 mod 5 and 2 mod 7?

    >>> solve_congruence((2, 3), (3, 5), (2, 7))
    (23, 105)
    >>> [23 % m for m in [3, 5, 7]]
    [2, 3, 2]

    If you prefer to work with all remainder in one list and
    all moduli in another, send the arguments like this:

    >>> solve_congruence(*zip((2, 3, 2), (3, 5, 7)))
    (23, 105)

    The moduli need not be co-prime; in this case there may or
    may not be a solution:

    >>> solve_congruence((2, 3), (4, 6)) is None
    True

    >>> solve_congruence((2, 3), (5, 6))
    (5, 6)

    The symmetric flag will make the result be within 1/2 of the modulus:

    >>> solve_congruence((2, 3), (5, 6), symmetric=True)
    (-1, 6)

    See Also
    ========

    crt : high level routine implementing the Chinese Remainder Theorem

    """

    def combine(c1, c2):
        """Return the tuple (a, m) which satisfies the requirement
        that n = a + i*m satisfy n = a1 + j*m1 and n = a2 = k*m2.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Method_of_successive_substitution
        """
        a1, m1 = c1
        a2, m2 = c2
        a, b, c = (m1, a2 - a1, m2)
        g = reduce(igcd, [a, b, c])
        a, b, c = [i // g for i in [a, b, c]]
        if a != 1:
            inv_a, _, g = igcdex(a, c)
            if g != 1:
                return None
            b *= inv_a
        a, m = (a1 + m1 * b, m1 * c)
        return (a, m)
    rm = remainder_modulus_pairs
    symmetric = hint.get('symmetric', False)
    if hint.get('check', True):
        rm = [(as_int(r), as_int(m)) for r, m in rm]
        uniq = {}
        for r, m in rm:
            r %= m
            if m in uniq:
                if r != uniq[m]:
                    return None
                continue
            uniq[m] = r
        rm = [(r, m) for m, r in uniq.items()]
        del uniq
        if all((isprime(m) for r, m in rm)):
            r, m = list(zip(*rm))
            return crt(m, r, symmetric=symmetric, check=False)
    rv = (0, 1)
    for rmi in rm:
        rv = combine(rv, rmi)
        if rv is None:
            break
        n, m = rv
        n = n % m
    else:
        if symmetric:
            return (symmetric_residue(n, m), m)
        return (n, m)