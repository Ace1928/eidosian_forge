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
def test_deprecated_iterables():
    from sympy.utilities.iterables import default_sort_key, ordered
    with warns_deprecated_sympy():
        assert list(ordered([y, x])) == [x, y]
    with warns_deprecated_sympy():
        assert sorted([y, x], key=default_sort_key) == [x, y]