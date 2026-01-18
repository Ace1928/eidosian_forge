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
def test_epsilon():
    Lorentz = TensorIndexType('Lorentz', dim=4, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    epsilon = Lorentz.epsilon
    p, q, r, s = tensor_heads('p,q,r,s', [Lorentz])
    t = epsilon(b, a, c, d)
    t1 = t.canon_bp()
    assert t1 == -epsilon(a, b, c, d)
    t = epsilon(c, b, d, a)
    t1 = t.canon_bp()
    assert t1 == epsilon(a, b, c, d)
    t = epsilon(c, a, d, b)
    t1 = t.canon_bp()
    assert t1 == -epsilon(a, b, c, d)
    t = epsilon(a, b, c, d) * p(-a) * q(-b)
    t1 = t.canon_bp()
    assert t1 == epsilon(c, d, a, b) * p(-a) * q(-b)
    t = epsilon(c, b, d, a) * p(-a) * q(-b)
    t1 = t.canon_bp()
    assert t1 == epsilon(c, d, a, b) * p(-a) * q(-b)
    t = epsilon(c, a, d, b) * p(-a) * q(-b)
    t1 = t.canon_bp()
    assert t1 == -epsilon(c, d, a, b) * p(-a) * q(-b)
    t = epsilon(c, a, d, b) * p(-a) * p(-b)
    t1 = t.canon_bp()
    assert t1 == 0
    t = epsilon(c, a, d, b) * p(-a) * q(-b) + epsilon(a, b, c, d) * p(-b) * q(-a)
    t1 = t.canon_bp()
    assert t1 == -2 * epsilon(c, d, a, b) * p(-a) * q(-b)
    Lorentz = TensorIndexType('Lorentz', dim=Integer(4), dummy_name='L')
    epsilon = Lorentz.epsilon
    assert isinstance(epsilon, TensorHead)