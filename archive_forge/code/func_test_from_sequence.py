from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_from_sequence():
    assert Permutation.from_sequence('SymPy') == Permutation(4)(0, 1, 3)
    assert Permutation.from_sequence('SymPy', key=lambda x: x.lower()) == Permutation(4)(0, 2)(1, 3)