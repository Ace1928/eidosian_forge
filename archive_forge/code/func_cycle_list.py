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
def cycle_list(k, n):
    """
    Returns the elements of the list ``range(n)`` shifted to the
    left by ``k`` (so the list starts with ``k`` (mod ``n``)).

    Examples
    ========

    >>> from sympy.crypto.crypto import cycle_list
    >>> cycle_list(3, 10)
    [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]

    """
    k = k % n
    return list(range(k, n)) + list(range(k))