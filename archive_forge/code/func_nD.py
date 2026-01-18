from math import prod
from collections import defaultdict
from typing import Tuple as tTuple
from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import (binomial,
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib
def nD(i=None, brute=None, *, n=None, m=None):
    """return the number of derangements for: ``n`` unique items, ``i``
    items (as a sequence or multiset), or multiplicities, ``m`` given
    as a sequence or multiset.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements as enum
    >>> from sympy.functions.combinatorial.numbers import nD

    A derangement ``d`` of sequence ``s`` has all ``d[i] != s[i]``:

    >>> set([''.join(i) for i in enum('abc')])
    {'bca', 'cab'}
    >>> nD('abc')
    2

    Input as iterable or dictionary (multiset form) is accepted:

    >>> assert nD([1, 2, 2, 3, 3, 3]) == nD({1: 1, 2: 2, 3: 3})

    By default, a brute-force enumeration and count of multiset permutations
    is only done if there are fewer than 9 elements. There may be cases when
    there is high multiplicity with few unique elements that will benefit
    from a brute-force enumeration, too. For this reason, the `brute`
    keyword (default None) is provided. When False, the brute-force
    enumeration will never be used. When True, it will always be used.

    >>> nD('1111222233', brute=True)
    44

    For convenience, one may specify ``n`` distinct items using the
    ``n`` keyword:

    >>> assert nD(n=3) == nD('abc') == 2

    Since the number of derangments depends on the multiplicity of the
    elements and not the elements themselves, it may be more convenient
    to give a list or multiset of multiplicities using keyword ``m``:

    >>> assert nD('abc') == nD(m=(1,1,1)) == nD(m={1:3}) == 2

    """
    from sympy.integrals.integrals import integrate
    from sympy.functions.special.polynomials import laguerre
    from sympy.abc import x

    def ok(x):
        if not isinstance(x, SYMPY_INTS):
            raise TypeError('expecting integer values')
        if x < 0:
            raise ValueError('value must not be negative')
        return True
    if (i, n, m).count(None) != 2:
        raise ValueError('enter only 1 of i, n, or m')
    if i is not None:
        if isinstance(i, SYMPY_INTS):
            raise TypeError('items must be a list or dictionary')
        if not i:
            return S.Zero
        if type(i) is not dict:
            s = list(i)
            ms = multiset(s)
        elif type(i) is dict:
            all((ok(_) for _ in i.values()))
            ms = {k: v for k, v in i.items() if v}
            s = None
        if not ms:
            return S.Zero
        N = sum(ms.values())
        counts = multiset(ms.values())
        nkey = len(ms)
    elif n is not None:
        ok(n)
        if not n:
            return S.Zero
        return subfactorial(n)
    elif m is not None:
        if isinstance(m, dict):
            all((ok(i) and ok(j) for i, j in m.items()))
            counts = {k: v for k, v in m.items() if k * v}
        elif iterable(m) or isinstance(m, str):
            m = list(m)
            all((ok(i) for i in m))
            counts = multiset([i for i in m if i])
        else:
            raise TypeError('expecting iterable')
        if not counts:
            return S.Zero
        N = sum((k * v for k, v in counts.items()))
        nkey = sum(counts.values())
        s = None
    big = int(max(counts))
    if big == 1:
        return subfactorial(nkey)
    nval = len(counts)
    if big * 2 > N:
        return S.Zero
    if big * 2 == N:
        if nkey == 2 and nval == 1:
            return S.One
        if nkey - 1 == big:
            return factorial(big)
    if N < 9 and brute is None or brute:
        if s is None:
            s = []
            i = 0
            for m, v in counts.items():
                for j in range(v):
                    s.extend([i] * m)
                    i += 1
        return Integer(sum((1 for i in multiset_derangements(s))))
    from sympy.functions.elementary.exponential import exp
    return Integer(abs(integrate(exp(-x) * Mul(*[laguerre(i, x) ** m for i, m in counts.items()]), (x, 0, oo))))