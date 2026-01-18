import string
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
from sympy.functions.combinatorial.numbers import _nT
from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer
from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x
def test_nC_nP_nT():
    from sympy.utilities.iterables import multiset_permutations, multiset_combinations, multiset_partitions, partitions, subsets, permutations
    from sympy.functions.combinatorial.numbers import nP, nC, nT, stirling, _stirling1, _stirling2, _multiset_histogram, _AOP_product
    from sympy.combinatorics.permutations import Permutation
    from sympy.core.random import choice
    c = string.ascii_lowercase
    for i in range(100):
        s = ''.join((choice(c) for i in range(7)))
        u = len(s) == len(set(s))
        try:
            tot = 0
            for i in range(8):
                check = nP(s, i)
                tot += check
                assert len(list(multiset_permutations(s, i))) == check
                if u:
                    assert nP(len(s), i) == check
            assert nP(s) == tot
        except AssertionError:
            print(s, i, 'failed perm test')
            raise ValueError()
    for i in range(100):
        s = ''.join((choice(c) for i in range(7)))
        u = len(s) == len(set(s))
        try:
            tot = 0
            for i in range(8):
                check = nC(s, i)
                tot += check
                assert len(list(multiset_combinations(s, i))) == check
                if u:
                    assert nC(len(s), i) == check
            assert nC(s) == tot
            if u:
                assert nC(len(s)) == tot
        except AssertionError:
            print(s, i, 'failed combo test')
            raise ValueError()
    for i in range(1, 10):
        tot = 0
        for j in range(1, i + 2):
            check = nT(i, j)
            assert check.is_Integer
            tot += check
            assert sum((1 for p in partitions(i, j, size=True) if p[0] == j)) == check
        assert nT(i) == tot
    for i in range(1, 10):
        tot = 0
        for j in range(1, i + 2):
            check = nT(range(i), j)
            tot += check
            assert len(list(multiset_partitions(list(range(i)), j))) == check
        assert nT(range(i)) == tot
    for i in range(100):
        s = ''.join((choice(c) for i in range(7)))
        u = len(s) == len(set(s))
        try:
            tot = 0
            for i in range(1, 8):
                check = nT(s, i)
                tot += check
                assert len(list(multiset_partitions(s, i))) == check
                if u:
                    assert nT(range(len(s)), i) == check
            if u:
                assert nT(range(len(s))) == tot
            assert nT(s) == tot
        except AssertionError:
            print(s, i, 'failed partition test')
            raise ValueError()
    assert [stirling(9, i, kind=1) for i in range(11)] == [0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1, 0]
    perms = list(permutations(range(4)))
    assert [sum((1 for p in perms if Permutation(p).cycles == i)) for i in range(5)] == [0, 6, 11, 6, 1] == [stirling(4, i, kind=1) for i in range(5)]
    assert [stirling(n, k, signed=1) for n in range(10) for k in range(1, n + 1)] == [1, -1, 1, 2, -3, 1, -6, 11, -6, 1, 24, -50, 35, -10, 1, -120, 274, -225, 85, -15, 1, 720, -1764, 1624, -735, 175, -21, 1, -5040, 13068, -13132, 6769, -1960, 322, -28, 1, 40320, -109584, 118124, -67284, 22449, -4536, 546, -36, 1]
    assert [stirling(n, k, kind=1) for n in range(10) for k in range(n + 1)] == [1, 0, 1, 0, 1, 1, 0, 2, 3, 1, 0, 6, 11, 6, 1, 0, 24, 50, 35, 10, 1, 0, 120, 274, 225, 85, 15, 1, 0, 720, 1764, 1624, 735, 175, 21, 1, 0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1, 0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1]
    assert [stirling(n, k, kind=2) for n in range(10) for k in range(n + 1)] == [1, 0, 1, 0, 1, 1, 0, 1, 3, 1, 0, 1, 7, 6, 1, 0, 1, 15, 25, 10, 1, 0, 1, 31, 90, 65, 15, 1, 0, 1, 63, 301, 350, 140, 21, 1, 0, 1, 127, 966, 1701, 1050, 266, 28, 1, 0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1]
    assert stirling(3, 4, kind=1) == stirling(3, 4, kind=1) == 0
    raises(ValueError, lambda: stirling(-2, 2))
    assert isinstance(_stirling1(6, 3), Integer)
    assert isinstance(_stirling2(6, 3), Integer)

    def delta(p):
        if len(p) == 1:
            return oo
        return min((abs(i[0] - i[1]) for i in subsets(p, 2)))
    parts = multiset_partitions(range(5), 3)
    d = 2
    assert sum((1 for p in parts if all((delta(i) >= d for i in p)))) == stirling(5, 3, d=d) == 7
    assert nC('abb', 2) == nC('aab', 2) == 2
    assert nP(3, 3, replacement=True) == nP('aabc', 3, replacement=True) == 27
    assert nP(3, 4) == 0
    assert nP('aabc', 5) == 0
    assert nC(4, 2, replacement=True) == nC('abcdd', 2, replacement=True) == len(list(multiset_combinations('aabbccdd', 2))) == 10
    assert nC('abcdd') == sum((nC('abcdd', i) for i in range(6))) == 24
    assert nC(list('abcdd'), 4) == 4
    assert nT('aaaa') == nT(4) == len(list(partitions(4))) == 5
    assert nT('aaab') == len(list(multiset_partitions('aaab'))) == 7
    assert nC('aabb' * 3, 3) == 4
    assert dict(_AOP_product((4, 1, 1, 1))) == {0: 1, 1: 4, 2: 7, 3: 8, 4: 8, 5: 7, 6: 4, 7: 1}
    t = (3, 9, 4, 6, 6, 5, 5, 2, 10, 4)
    assert sum((_AOP_product(t)[i] for i in range(55))) == 58212000
    raises(ValueError, lambda: _multiset_histogram({1: 'a'}))