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
def test_multiset_partitions():
    A = [0, 1, 2, 3, 4]
    assert list(multiset_partitions(A, 5)) == [[[0], [1], [2], [3], [4]]]
    assert len(list(multiset_partitions(A, 4))) == 10
    assert len(list(multiset_partitions(A, 3))) == 25
    assert list(multiset_partitions([1, 1, 1, 2, 2], 2)) == [[[1, 1, 1, 2], [2]], [[1, 1, 1], [2, 2]], [[1, 1, 2, 2], [1]], [[1, 1, 2], [1, 2]], [[1, 1], [1, 2, 2]]]
    assert list(multiset_partitions([1, 1, 2, 2], 2)) == [[[1, 1, 2], [2]], [[1, 1], [2, 2]], [[1, 2, 2], [1]], [[1, 2], [1, 2]]]
    assert list(multiset_partitions([1, 2, 3, 4], 2)) == [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]], [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]], [[1], [2, 3, 4]]]
    assert list(multiset_partitions([1, 2, 2], 2)) == [[[1, 2], [2]], [[1], [2, 2]]]
    assert list(multiset_partitions(3)) == [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]], [[0], [1], [2]]]
    assert list(multiset_partitions(3, 2)) == [[[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]]]
    assert list(multiset_partitions([1] * 3, 2)) == [[[1], [1, 1]]]
    assert list(multiset_partitions([1] * 3)) == [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]
    a = [3, 2, 1]
    assert list(multiset_partitions(a)) == list(multiset_partitions(sorted(a)))
    assert list(multiset_partitions(a, 5)) == []
    assert list(multiset_partitions(a, 1)) == [[[1, 2, 3]]]
    assert list(multiset_partitions(a + [4], 5)) == []
    assert list(multiset_partitions(a + [4], 1)) == [[[1, 2, 3, 4]]]
    assert list(multiset_partitions(2, 5)) == []
    assert list(multiset_partitions(2, 1)) == [[[0, 1]]]
    assert list(multiset_partitions('a')) == [[['a']]]
    assert list(multiset_partitions('a', 2)) == []
    assert list(multiset_partitions('ab')) == [[['a', 'b']], [['a'], ['b']]]
    assert list(multiset_partitions('ab', 1)) == [[['a', 'b']]]
    assert list(multiset_partitions('aaa', 1)) == [['aaa']]
    assert list(multiset_partitions([1, 1], 1)) == [[[1, 1]]]
    ans = [('mpsyy',), ('mpsy', 'y'), ('mps', 'yy'), ('mps', 'y', 'y'), ('mpyy', 's'), ('mpy', 'sy'), ('mpy', 's', 'y'), ('mp', 'syy'), ('mp', 'sy', 'y'), ('mp', 's', 'yy'), ('mp', 's', 'y', 'y'), ('msyy', 'p'), ('msy', 'py'), ('msy', 'p', 'y'), ('ms', 'pyy'), ('ms', 'py', 'y'), ('ms', 'p', 'yy'), ('ms', 'p', 'y', 'y'), ('myy', 'ps'), ('myy', 'p', 's'), ('my', 'psy'), ('my', 'ps', 'y'), ('my', 'py', 's'), ('my', 'p', 'sy'), ('my', 'p', 's', 'y'), ('m', 'psyy'), ('m', 'psy', 'y'), ('m', 'ps', 'yy'), ('m', 'ps', 'y', 'y'), ('m', 'pyy', 's'), ('m', 'py', 'sy'), ('m', 'py', 's', 'y'), ('m', 'p', 'syy'), ('m', 'p', 'sy', 'y'), ('m', 'p', 's', 'yy'), ('m', 'p', 's', 'y', 'y')]
    assert [tuple((''.join(part) for part in p)) for p in multiset_partitions('sympy')] == ans
    factorings = [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3], [6, 2, 2], [2, 2, 2, 3]]
    assert [factoring_visitor(p, [2, 3]) for p in multiset_partitions_taocp([3, 1])] == factorings