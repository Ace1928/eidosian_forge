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
def test_contract_metric1():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    g = Lorentz.metric
    p = TensorHead('p', [Lorentz])
    t = g(a, b) * p(-b)
    t1 = t.contract_metric(g)
    assert t1 == p(a)
    A, B = tensor_heads('A,B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t1 = A(a, b) * B(-b, c) * g(d, e)
    t2 = t1.contract_metric(g)
    assert t1 == t2
    t1 = A(a, b) * B(-b, c) * g(-d, d)
    t2 = t1.contract_metric(g)
    assert t2 == D * A(a, d) * B(-d, c)
    t1 = A(a, b) * B(-b, -c) * g(c, d)
    t2 = t1.contract_metric(g)
    assert t2 == A(a, c) * B(-c, d)
    t1 = A(a, b) * B(-b, -c) * g(c, -a)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, b) * B(-b, -a))
    t1 = A(a, b) * B(-b, -c) * g(c, d) * g(-a, -d)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, b) * B(-b, -a))
    t1 = A(a, b) * g(-a, -b)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, -a))
    assert not t2.free
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    g = Lorentz.metric
    assert _is_equal(g(a, -a).contract_metric(g), Lorentz.dim)