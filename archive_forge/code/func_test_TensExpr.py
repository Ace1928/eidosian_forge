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
def test_TensExpr():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    g = Lorentz.metric
    A, B = tensor_heads('A B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    raises(ValueError, lambda: g(c, d) / g(a, b))
    raises(ValueError, lambda: S.One / g(a, b))
    raises(ValueError, lambda: (A(c, d) + g(c, d)) / g(a, b))
    raises(ValueError, lambda: S.One / (A(c, d) + g(c, d)))
    raises(ValueError, lambda: A(a, b) + A(a, c))
    with warns_deprecated_sympy():
        raises(ValueError, lambda: A(a, b) ** 2)
    raises(NotImplementedError, lambda: 2 ** A(a, b))
    raises(NotImplementedError, lambda: abs(A(a, b)))