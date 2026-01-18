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
def nT(n, k=None):
    """Return the number of ``k``-sized partitions of ``n`` items.

    Possible values for ``n``:

        integer - ``n`` identical items

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    Note: the convention for ``nT`` is different than that of ``nC`` and
    ``nP`` in that
    here an integer indicates ``n`` *identical* items instead of a set of
    length ``n``; this is in keeping with the ``partitions`` function which
    treats its integer-``n`` input like a list of ``n`` 1s. One can use
    ``range(n)`` for ``n`` to indicate ``n`` distinct items.

    If ``k`` is None then the total number of ways to partition the elements
    represented in ``n`` will be returned.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nT

    Partitions of the given multiset:

    >>> [nT('aabbc', i) for i in range(1, 7)]
    [1, 8, 11, 5, 1, 0]
    >>> nT('aabbc') == sum(_)
    True

    >>> [nT("mississippi", i) for i in range(1, 12)]
    [1, 74, 609, 1521, 1768, 1224, 579, 197, 50, 9, 1]

    Partitions when all items are identical:

    >>> [nT(5, i) for i in range(1, 6)]
    [1, 2, 2, 1, 1]
    >>> nT('1'*5) == sum(_)
    True

    When all items are different:

    >>> [nT(range(5), i) for i in range(1, 6)]
    [1, 15, 25, 10, 1]
    >>> nT(range(5)) == sum(_)
    True

    Partitions of an integer expressed as a sum of positive integers:

    >>> from sympy import partition
    >>> partition(4)
    5
    >>> nT(4, 1) + nT(4, 2) + nT(4, 3) + nT(4, 4)
    5
    >>> nT('1'*4)
    5

    See Also
    ========
    sympy.utilities.iterables.partitions
    sympy.utilities.iterables.multiset_partitions
    sympy.functions.combinatorial.numbers.partition

    References
    ==========

    .. [1] https://web.archive.org/web/20210507012732/https://teaching.csse.uwa.edu.au/units/CITS7209/partition.pdf

    """
    if isinstance(n, SYMPY_INTS):
        if k is None:
            return partition(n)
        if isinstance(k, SYMPY_INTS):
            n = as_int(n)
            k = as_int(k)
            return Integer(_nT(n, k))
    if not isinstance(n, _MultisetHistogram):
        try:
            u = len(set(n))
            if u <= 1:
                return nT(len(n), k)
            elif u == len(n):
                n = range(u)
            raise TypeError
        except TypeError:
            n = _multiset_histogram(n)
    N = n[_N]
    if k is None and N == 1:
        return 1
    if k in (1, N):
        return 1
    if k == 2 or (N == 2 and k is None):
        m, r = divmod(N, 2)
        rv = sum((nC(n, i) for i in range(1, m + 1)))
        if not r:
            rv -= nC(n, m) // 2
        if k is None:
            rv += 1
        return rv
    if N == n[_ITEMS]:
        if k is None:
            return bell(N)
        return stirling(N, k)
    m = MultisetPartitionTraverser()
    if k is None:
        return m.count_partitions(n[_M])
    tot = 0
    for discard in m.enum_range(n[_M], k - 1, k):
        tot += 1
    return tot