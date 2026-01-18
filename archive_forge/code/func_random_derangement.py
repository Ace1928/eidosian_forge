from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def random_derangement(t, choice=None, strict=True):
    """Return a list of elements in which none are in the same positions
    as they were originally. If an element fills more than half of the positions
    then an error will be raised since no derangement is possible. To obtain
    a derangement of as many items as possible--with some of the most numerous
    remaining in their original positions--pass `strict=False`. To produce a
    pseudorandom derangment, pass a pseudorandom selector like `choice` (see
    below).

    Examples
    ========

    >>> from sympy.utilities.iterables import random_derangement
    >>> t = 'SymPy: a CAS in pure Python'
    >>> d = random_derangement(t)
    >>> all(i != j for i, j in zip(d, t))
    True

    A predictable result can be obtained by using a pseudorandom
    generator for the choice:

    >>> from sympy.core.random import seed, choice as c
    >>> seed(1)
    >>> d = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert len(set(d)) != 1  # we got different values

    By reseeding, the same sequence can be obtained:

    >>> seed(1)
    >>> d2 = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert d == d2
    """
    if choice is None:
        import secrets
        choice = secrets.choice

    def shuffle(rv):
        """Knuth shuffle"""
        for i in range(len(rv) - 1, 0, -1):
            x = choice(rv[:i + 1])
            j = rv.index(x)
            rv[i], rv[j] = (rv[j], rv[i])

    def pick(rv, n):
        """shuffle rv and return the first n values
        """
        shuffle(rv)
        return rv[:n]
    ms = multiset(t)
    tot = len(t)
    ms = sorted(ms.items(), key=lambda x: x[1])
    M, mx = ms[-1]
    n = len(t)
    xs = 2 * mx - tot
    if xs > 0:
        if strict:
            raise ValueError('no derangement possible')
        opts = [i for i, c in enumerate(t) if c == ms[-1][0]]
        pick(opts, xs)
        stay = sorted(opts[:xs])
        rv = list(t)
        for i in reversed(stay):
            rv.pop(i)
        rv = random_derangement(rv, choice)
        for i in stay:
            rv.insert(i, ms[-1][0])
        return ''.join(rv) if type(t) is str else rv
    if n == len(ms):
        rv = list(t)
        while True:
            shuffle(rv)
            if all((i != j for i, j in zip(rv, t))):
                break
    else:
        rv = [None] * n
        while True:
            j = 0
            while j > -len(ms):
                j -= 1
                e, c = ms[j]
                opts = [i for i in range(n) if rv[i] is None and t[i] != e]
                if len(opts) < c:
                    for i in range(n):
                        rv[i] = None
                    break
                pick(opts, c)
                for i in range(c):
                    rv[opts[i]] = e
            else:
                return rv
    return rv