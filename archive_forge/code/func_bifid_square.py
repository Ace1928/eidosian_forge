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
def bifid_square(key):
    """Return characters of ``key`` arranged in a square.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...    bifid_square, AZ, padded_key, bifid5)
    >>> bifid_square(AZ().replace('J', ''))
    Matrix([
    [A, B, C, D, E],
    [F, G, H, I, K],
    [L, M, N, O, P],
    [Q, R, S, T, U],
    [V, W, X, Y, Z]])

    >>> bifid_square(padded_key(AZ('gold bug!'), bifid5))
    Matrix([
    [G, O, L, D, B],
    [U, A, C, E, F],
    [H, I, K, M, N],
    [P, Q, R, S, T],
    [V, W, X, Y, Z]])

    See Also
    ========

    padded_key

    """
    A = ''.join(uniq(''.join(key)))
    n = len(A) ** 0.5
    if n != int(n):
        raise ValueError('Length of alphabet (%s) is not a square number.' % len(A))
    n = int(n)
    f = lambda i, j: Symbol(A[n * i + j])
    rv = Matrix(n, n, f)
    return rv