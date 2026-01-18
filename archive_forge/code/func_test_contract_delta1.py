from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_contract_delta1():
    n = Symbol('n')
    Color = TensorIndexType('Color', dim=n, dummy_name='C')
    a, b, c, d, e, f = tensor_indices('a,b,c,d,e,f', Color)
    delta = Color.delta

    def idn(a, b, d, c):
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a, c) * delta(d, b)

    def T(a, b, d, c):
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a, b) * delta(d, c)

    def P1(a, b, c, d):
        return idn(a, b, c, d) - 1 / n * T(a, b, c, d)

    def P2(a, b, c, d):
        return 1 / n * T(a, b, c, d)
    t = P1(a, -b, e, -f) * P1(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert canon_bp(t1 - P1(a, -b, d, -c)) == 0
    t = P2(a, -b, e, -f) * P2(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert t1 == P2(a, -b, d, -c)
    t = P1(a, -b, e, -f) * P2(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert t1 == 0
    t = P1(a, -b, b, -a)
    t1 = t.contract_delta(delta)
    assert t1.equals(n ** 2 - 1)