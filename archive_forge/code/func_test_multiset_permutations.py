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
def test_multiset_permutations():
    ans = ['abby', 'abyb', 'aybb', 'baby', 'bayb', 'bbay', 'bbya', 'byab', 'byba', 'yabb', 'ybab', 'ybba']
    assert [''.join(i) for i in multiset_permutations('baby')] == ans
    assert [''.join(i) for i in multiset_permutations(multiset('baby'))] == ans
    assert list(multiset_permutations([0, 0, 0], 2)) == [[0, 0]]
    assert list(multiset_permutations([0, 2, 1], 2)) == [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    assert len(list(multiset_permutations('a', 0))) == 1
    assert len(list(multiset_permutations('a', 3))) == 0
    for nul in ([], {}, ''):
        assert list(multiset_permutations(nul)) == [[]]
    assert list(multiset_permutations(nul, 0)) == [[]]
    assert list(multiset_permutations(nul, 1)) == []
    assert list(multiset_permutations(nul, -1)) == []

    def test():
        for i in range(1, 7):
            print(i)
            for p in multiset_permutations([0, 0, 1, 0, 1], i):
                print(p)
    assert capture(lambda: test()) == dedent('        1\n        [0]\n        [1]\n        2\n        [0, 0]\n        [0, 1]\n        [1, 0]\n        [1, 1]\n        3\n        [0, 0, 0]\n        [0, 0, 1]\n        [0, 1, 0]\n        [0, 1, 1]\n        [1, 0, 0]\n        [1, 0, 1]\n        [1, 1, 0]\n        4\n        [0, 0, 0, 1]\n        [0, 0, 1, 0]\n        [0, 0, 1, 1]\n        [0, 1, 0, 0]\n        [0, 1, 0, 1]\n        [0, 1, 1, 0]\n        [1, 0, 0, 0]\n        [1, 0, 0, 1]\n        [1, 0, 1, 0]\n        [1, 1, 0, 0]\n        5\n        [0, 0, 0, 1, 1]\n        [0, 0, 1, 0, 1]\n        [0, 0, 1, 1, 0]\n        [0, 1, 0, 0, 1]\n        [0, 1, 0, 1, 0]\n        [0, 1, 1, 0, 0]\n        [1, 0, 0, 0, 1]\n        [1, 0, 0, 1, 0]\n        [1, 0, 1, 0, 0]\n        [1, 1, 0, 0, 0]\n        6\n')
    raises(ValueError, lambda: list(multiset_permutations({0: 3, 1: -1})))