from textwrap import dedent
from itertools import islice, product
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices.dense import Matrix
from sympy.combinatorics import RGS_enum, RGS_unrank, Permutation
from sympy.utilities.iterables import (
from sympy.utilities.enumerative import (
from sympy.core.singleton import S
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_multiset_combinations():
    ans = ['iii', 'iim', 'iip', 'iis', 'imp', 'ims', 'ipp', 'ips', 'iss', 'mpp', 'mps', 'mss', 'pps', 'pss', 'sss']
    assert [''.join(i) for i in list(multiset_combinations('mississippi', 3))] == ans
    M = multiset('mississippi')
    assert [''.join(i) for i in list(multiset_combinations(M, 3))] == ans
    assert [''.join(i) for i in multiset_combinations(M, 30)] == []
    assert list(multiset_combinations([[1], [2, 3]], 2)) == [[[1], [2, 3]]]
    assert len(list(multiset_combinations('a', 3))) == 0
    assert len(list(multiset_combinations('a', 0))) == 1
    assert list(multiset_combinations('abc', 1)) == [['a'], ['b'], ['c']]
    raises(ValueError, lambda: list(multiset_combinations({0: 3, 1: -1}, 2)))