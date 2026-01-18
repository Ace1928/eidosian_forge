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
def test_kbins():
    assert len(list(kbins('1123', 2, ordered=1))) == 24
    assert len(list(kbins('1123', 2, ordered=11))) == 36
    assert len(list(kbins('1123', 2, ordered=10))) == 10
    assert len(list(kbins('1123', 2, ordered=0))) == 5
    assert len(list(kbins('1123', 2, ordered=None))) == 3

    def test1():
        for orderedval in [None, 0, 1, 10, 11]:
            print('ordered =', orderedval)
            for p in kbins([0, 0, 1], 2, ordered=orderedval):
                print('   ', p)
    assert capture(lambda: test1()) == dedent('        ordered = None\n            [[0], [0, 1]]\n            [[0, 0], [1]]\n        ordered = 0\n            [[0, 0], [1]]\n            [[0, 1], [0]]\n        ordered = 1\n            [[0], [0, 1]]\n            [[0], [1, 0]]\n            [[1], [0, 0]]\n        ordered = 10\n            [[0, 0], [1]]\n            [[1], [0, 0]]\n            [[0, 1], [0]]\n            [[0], [0, 1]]\n        ordered = 11\n            [[0], [0, 1]]\n            [[0, 0], [1]]\n            [[0], [1, 0]]\n            [[0, 1], [0]]\n            [[1], [0, 0]]\n            [[1, 0], [0]]\n')

    def test2():
        for orderedval in [None, 0, 1, 10, 11]:
            print('ordered =', orderedval)
            for p in kbins(list(range(3)), 2, ordered=orderedval):
                print('   ', p)
    assert capture(lambda: test2()) == dedent('        ordered = None\n            [[0], [1, 2]]\n            [[0, 1], [2]]\n        ordered = 0\n            [[0, 1], [2]]\n            [[0, 2], [1]]\n            [[0], [1, 2]]\n        ordered = 1\n            [[0], [1, 2]]\n            [[0], [2, 1]]\n            [[1], [0, 2]]\n            [[1], [2, 0]]\n            [[2], [0, 1]]\n            [[2], [1, 0]]\n        ordered = 10\n            [[0, 1], [2]]\n            [[2], [0, 1]]\n            [[0, 2], [1]]\n            [[1], [0, 2]]\n            [[0], [1, 2]]\n            [[1, 2], [0]]\n        ordered = 11\n            [[0], [1, 2]]\n            [[0, 1], [2]]\n            [[0], [2, 1]]\n            [[0, 2], [1]]\n            [[1], [0, 2]]\n            [[1, 0], [2]]\n            [[1], [2, 0]]\n            [[1, 2], [0]]\n            [[2], [0, 1]]\n            [[2, 0], [1]]\n            [[2], [1, 0]]\n            [[2, 1], [0]]\n')