from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet
from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin
from itertools import product
def test_issue_22021():
    from sympy.calculus.accumulationbounds import AccumBounds
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
    L = TensorIndexType('L')
    i = tensor_indices('i', L)
    A, B = tensor_heads('A B', [L])
    e = A(i) + B(i)
    assert -e == -1 * e
    e = zoo + x
    assert -e == -1 * e
    a = AccumBounds(1, 2)
    e = a + x
    assert -e == -1 * e
    for args in permutations((zoo, a, x)):
        e = Add(*args, evaluate=False)
        assert -e == -1 * e
    assert 2 * Add(1, x, x, evaluate=False) == 4 * x + 2