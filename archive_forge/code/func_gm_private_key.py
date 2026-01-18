from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import igcdex, mod_inverse, igcd, Rational
from sympy.core.random import _randrange, _randint
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import gcd, Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
def gm_private_key(p, q, a=None):
    """
    Check if ``p`` and ``q`` can be used as private keys for
    the Goldwasser-Micali encryption. The method works
    roughly as follows.

    Explanation
    ===========

    #. Pick two large primes $p$ and $q$.
    #. Call their product $N$.
    #. Given a message as an integer $i$, write $i$ in its bit representation $b_0, \\dots, b_n$.
    #. For each $k$,

     if $b_k = 0$:
        let $a_k$ be a random square
        (quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``
     if $b_k = 1$:
        let $a_k$ be a random non-square
        (non-quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``

    returns $\\left[a_1, a_2, \\dots\\right]$

    $b_k$ can be recovered by checking whether or not
    $a_k$ is a residue. And from the $b_k$'s, the message
    can be reconstructed.

    The idea is that, while ``jacobi_symbol(a, p*q)``
    can be easily computed (and when it is equal to $-1$ will
    tell you that $a$ is not a square mod $p q$), quadratic
    residuosity modulo a composite number is hard to compute
    without knowing its factorization.

    Moreover, approximately half the numbers coprime to $p q$ have
    :func:`~.jacobi_symbol` equal to $1$ . And among those, approximately half
    are residues and approximately half are not. This maximizes the
    entropy of the code.

    Parameters
    ==========

    p, q, a
        Initialization variables.

    Returns
    =======

    tuple : (p, q)
        The input value ``p`` and ``q``.

    Raises
    ======

    ValueError
        If ``p`` and ``q`` are not distinct odd primes.

    """
    if p == q:
        raise ValueError('expected distinct primes, got two copies of %i' % p)
    elif not isprime(p) or not isprime(q):
        raise ValueError('first two arguments must be prime, got %i of %i' % (p, q))
    elif p == 2 or q == 2:
        raise ValueError('first two arguments must not be even, got %i of %i' % (p, q))
    return (p, q)