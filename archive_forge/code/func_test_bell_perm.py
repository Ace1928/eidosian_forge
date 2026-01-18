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
def test_bell_perm():
    assert [len(set(generate_bell(i))) for i in range(1, 7)] == [factorial(i) for i in range(1, 7)]
    assert list(generate_bell(3)) == [(0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    for n in range(1, 5):
        p = Permutation(range(n))
        b = generate_bell(n)
        for bi in b:
            assert bi == tuple(p.array_form)
            p = p.next_trotterjohnson()
    raises(ValueError, lambda: list(generate_bell(0)))