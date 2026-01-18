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
def test_necklaces():

    def count(n, k, f):
        return len(list(necklaces(n, k, f)))
    m = []
    for i in range(1, 8):
        m.append((i, count(i, 2, 0), count(i, 2, 1), count(i, 3, 1)))
    assert Matrix(m) == Matrix([[1, 2, 2, 3], [2, 3, 3, 6], [3, 4, 4, 10], [4, 6, 6, 21], [5, 8, 8, 39], [6, 14, 13, 92], [7, 20, 18, 198]])