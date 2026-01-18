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
def nC(n, k=None, replacement=False):
    """Return the number of combinations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all combinations of length 0
    through the number of items represented in ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' sets of 2 would include 'aa',
    'ab', and 'bb'.) The multiplicity of elements in ``n`` is ignored when
    ``replacement`` is True but the total number of elements is considered
    since no element can appear more times than the number of elements in
    ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nC
    >>> from sympy.utilities.iterables import multiset_combinations
    >>> nC(3, 2)
    3
    >>> nC('abc', 2)
    3
    >>> nC('aab', 2)
    2

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nC('aabc', replacement=True)
    35
    >>> [len(list(multiset_combinations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 6, 10, 15]
    >>> sum(_)
    35

    If there are ``k`` items with multiplicities ``m_1, m_2, ..., m_k``
    then the total of all combinations of length 0 through ``k`` is the
    product, ``(m_1 + 1)*(m_2 + 1)*...*(m_k + 1)``. When the multiplicity
    of each item is 1 (i.e., k unique items) then there are 2**k
    combinations. For example, if there are 4 unique items, the total number
    of combinations is 16:

    >>> sum(nC(4, i) for i in range(5))
    16

    See Also
    ========

    sympy.utilities.iterables.multiset_combinations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combination
    .. [2] https://tinyurl.com/cep849r

    """
    if isinstance(n, SYMPY_INTS):
        if k is None:
            if not replacement:
                return 2 ** n
            return sum((nC(n, i, replacement) for i in range(n + 1)))
        if k < 0:
            raise ValueError('k cannot be negative')
        if replacement:
            return binomial(n + k - 1, k)
        return binomial(n, k)
    if isinstance(n, _MultisetHistogram):
        N = n[_N]
        if k is None:
            if not replacement:
                return prod((m + 1 for m in n[_M]))
            return sum((nC(n, i, replacement) for i in range(N + 1)))
        elif replacement:
            return nC(n[_ITEMS], k, replacement)
        elif k in (1, N - 1):
            return n[_ITEMS]
        elif k in (0, N):
            return 1
        return _AOP_product(tuple(n[_M]))[k]
    else:
        return nC(_multiset_histogram(n), k, replacement)