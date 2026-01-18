from sympy.ntheory import sieve, isprime
from sympy.core.numbers import mod_inverse
from sympy.core.power import integer_log
from sympy.utilities.misc import as_int
import random
def ecm(n, B1=10000, B2=100000, max_curve=200, seed=1234):
    """Performs factorization using Lenstra's Elliptic curve method.

    This function repeatedly calls `ecm_one_factor` to compute the factors
    of n. First all the small factors are taken out using trial division.
    Then `ecm_one_factor` is used to compute one factor at a time.

    Parameters
    ==========

    n : Number to be Factored
    B1 : Stage 1 Bound
    B2 : Stage 2 Bound
    max_curve : Maximum number of curves generated
    seed : Initialize pseudorandom generator

    Examples
    ========

    >>> from sympy.ntheory import ecm
    >>> ecm(25645121643901801)
    {5394769, 4753701529}
    >>> ecm(9804659461513846513)
    {4641991, 2112166839943}
    """
    _factors = set()
    for prime in sieve.primerange(1, 100000):
        if n % prime == 0:
            _factors.add(prime)
            while n % prime == 0:
                n //= prime
    rgen.seed(seed)
    while n > 1:
        try:
            factor = _ecm_one_factor(n, B1, B2, max_curve)
        except ValueError:
            raise ValueError('Increase the bounds')
        _factors.add(factor)
        n //= factor
    factors = set()
    for factor in _factors:
        if isprime(factor):
            factors.add(factor)
            continue
        factors |= ecm(factor)
    return factors