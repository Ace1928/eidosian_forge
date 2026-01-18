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
def test_add1():
    assert TensAdd().args == ()
    assert TensAdd().doit() == 0
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, d0, d1, i, j, k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t1 = A(b, -d0) * B(d0, a)
    assert TensAdd(t1).equals(t1)
    t2a = B(d0, a) + A(d0, a)
    t2 = A(b, -d0) * t2a
    assert str(t2) == 'A(b, -L_0)*(A(L_0, a) + B(L_0, a))'
    t2 = t2.expand()
    assert str(t2) == 'A(b, -L_0)*A(L_0, a) + A(b, -L_0)*B(L_0, a)'
    t2 = t2.canon_bp()
    assert str(t2) == 'A(a, L_0)*A(b, -L_0) + A(b, L_0)*B(a, -L_0)'
    t2b = t2 + t1
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + A(b, -L_0)*B(L_0, a) + A(b, L_0)*B(a, -L_0)'
    t2b = t2b.canon_bp()
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + 2*A(b, L_0)*B(a, -L_0)'
    p, q, r = tensor_heads('p,q,r', [Lorentz])
    t = q(d0) * 2
    assert str(t) == '2*q(d0)'
    t = 2 * q(d0)
    assert str(t) == '2*q(d0)'
    t1 = p(d0) + 2 * q(d0)
    assert str(t1) == '2*q(d0) + p(d0)'
    t2 = p(-d0) + 2 * q(-d0)
    assert str(t2) == '2*q(-d0) + p(-d0)'
    t1 = p(d0)
    t3 = t1 * t2
    assert str(t3) == 'p(L_0)*(2*q(-L_0) + p(-L_0))'
    t3 = t3.expand()
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    t3 = t2 * t1
    t3 = t3.expand()
    assert str(t3) == 'p(-L_0)*p(L_0) + 2*q(-L_0)*p(L_0)'
    t3 = t3.canon_bp()
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    t1 = p(d0) + 2 * q(d0)
    t3 = t1 * t2
    t3 = t3.canon_bp()
    assert str(t3) == 'p(L_0)*p(-L_0) + 4*p(L_0)*q(-L_0) + 4*q(L_0)*q(-L_0)'
    t1 = p(d0) - 2 * q(d0)
    assert str(t1) == '-2*q(d0) + p(d0)'
    t2 = p(-d0) + 2 * q(-d0)
    t3 = t1 * t2
    t3 = t3.canon_bp()
    assert t3 == p(d0) * p(-d0) - 4 * q(d0) * q(-d0)
    t = p(i) * p(j) * (p(k) + q(k)) + p(i) * (p(j) + q(j)) * (p(k) - 3 * q(k))
    t = t.canon_bp()
    assert t == 2 * p(i) * p(j) * p(k) - 2 * p(i) * p(j) * q(k) + p(i) * p(k) * q(j) - 3 * p(i) * q(j) * q(k)
    t1 = (p(i) + q(i) + 2 * r(i)) * (p(j) - q(j))
    t2 = (p(j) + q(j) + 2 * r(j)) * (p(i) - q(i))
    t = t1 + t2
    t = t.canon_bp()
    assert t == 2 * p(i) * p(j) + 2 * p(i) * r(j) + 2 * p(j) * r(i) - 2 * q(i) * q(j) - 2 * q(i) * r(j) - 2 * q(j) * r(i)
    t = p(i) * q(j) / 2
    assert 2 * t == p(i) * q(j)
    t = (p(i) + q(i)) / 2
    assert 2 * t == p(i) + q(i)
    t = S.One - p(i) * p(-i)
    t = t.canon_bp()
    tz1 = t + p(-j) * p(j)
    assert tz1 != 1
    tz1 = tz1.canon_bp()
    assert tz1.equals(1)
    t = S.One + p(i) * p(-i)
    assert (t - p(-j) * p(j)).canon_bp().equals(1)
    t = A(a, b) + B(a, b)
    assert t.rank == 2
    t1 = t - A(a, b) - B(a, b)
    assert t1 == 0
    t = 1 - (A(a, -a) + B(a, -a))
    t1 = 1 + (A(a, -a) + B(a, -a))
    assert (t + t1).expand().equals(2)
    t2 = 1 + A(a, -a)
    assert t1 != t2
    assert t2 != TensMul.from_data(0, [], [], [])
    assert TensAdd(p(a), TensMul(0, p(a))).doit() == p(a)