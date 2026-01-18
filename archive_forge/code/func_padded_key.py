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
def padded_key(key, symbols):
    """Return a string of the distinct characters of ``symbols`` with
    those of ``key`` appearing first. A ValueError is raised if
    a) there are duplicate characters in ``symbols`` or
    b) there are characters in ``key`` that are  not in ``symbols``.

    Examples
    ========

    >>> from sympy.crypto.crypto import padded_key
    >>> padded_key('PUPPY', 'OPQRSTUVWXY')
    'PUYOQRSTVWX'
    >>> padded_key('RSA', 'ARTIST')
    Traceback (most recent call last):
    ...
    ValueError: duplicate characters in symbols: T

    """
    syms = list(uniq(symbols))
    if len(syms) != len(symbols):
        extra = ''.join(sorted({i for i in symbols if symbols.count(i) > 1}))
        raise ValueError('duplicate characters in symbols: %s' % extra)
    extra = set(key) - set(syms)
    if extra:
        raise ValueError('characters in key but not symbols: %s' % ''.join(sorted(extra)))
    key0 = ''.join(list(uniq(key)))
    return key0 + translate(''.join(syms), None, key0)