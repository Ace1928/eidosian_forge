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
def test_add2():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m, n, p, q = tensor_indices('m,n,p,q', Lorentz)
    R = TensorHead('R', [Lorentz] * 4, TensorSymmetry.riemann())
    A = TensorHead('A', [Lorentz] * 3, TensorSymmetry.fully_symmetric(-3))
    t1 = 2 * R(m, n, p, q) - R(m, q, n, p) + R(m, p, n, q)
    t2 = t1 * A(-n, -p, -q)
    t2 = t2.canon_bp()
    assert t2 == 0
    t1 = Rational(2, 3) * R(m, n, p, q) - Rational(1, 3) * R(m, q, n, p) + Rational(1, 3) * R(m, p, n, q)
    t2 = t1 * A(-n, -p, -q)
    t2 = t2.canon_bp()
    assert t2 == 0
    t = A(m, -m, n) + A(n, p, -p)
    t = t.canon_bp()
    assert t == 0