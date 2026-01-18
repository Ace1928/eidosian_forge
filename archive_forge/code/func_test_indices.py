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
def test_indices():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    assert a.tensor_index_type == Lorentz
    assert a != -a
    A, B = tensor_heads('A B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(a, b) * B(-b, c)
    indices = t.get_indices()
    L_0 = TensorIndex('L_0', Lorentz)
    assert indices == [a, L_0, -L_0, c]
    raises(ValueError, lambda: tensor_indices(3, Lorentz))
    raises(ValueError, lambda: A(a, b, c))
    A = TensorHead('A', [Lorentz, Lorentz])
    assert A('a', 'b') == A(TensorIndex('a', Lorentz), TensorIndex('b', Lorentz))
    assert A('a', '-b') == A(TensorIndex('a', Lorentz), TensorIndex('b', Lorentz, is_up=False))
    assert A('a', TensorIndex('b', Lorentz)) == A(TensorIndex('a', Lorentz), TensorIndex('b', Lorentz))