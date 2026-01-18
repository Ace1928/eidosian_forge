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
def test_iproduct():
    assert list(iproduct()) == [()]
    assert list(iproduct([])) == []
    assert list(iproduct([1, 2, 3])) == [(1,), (2,), (3,)]
    assert sorted(iproduct([1, 2], [3, 4, 5])) == [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    assert sorted(iproduct([0, 1], [0, 1], [0, 1])) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    assert iterable(iproduct(S.Integers)) is True
    assert iterable(iproduct(S.Integers, S.Integers)) is True
    assert (3,) in iproduct(S.Integers)
    assert (4, 5) in iproduct(S.Integers, S.Integers)
    assert (1, 2, 3) in iproduct(S.Integers, S.Integers, S.Integers)
    triples = set(islice(iproduct(S.Integers, S.Integers, S.Integers), 1000))
    for n1, n2, n3 in triples:
        assert isinstance(n1, Integer)
        assert isinstance(n2, Integer)
        assert isinstance(n3, Integer)
    for t in set(product(*[range(-2, 3)] * 3)):
        assert t in iproduct(S.Integers, S.Integers, S.Integers)