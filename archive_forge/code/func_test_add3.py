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
def test_add3():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i0, i1 = tensor_indices('i0:2', Lorentz)
    E, px, py, pz = symbols('E px py pz')
    A = TensorHead('A', [Lorentz])
    B = TensorHead('B', [Lorentz])
    expr1 = A(i0) * A(-i0) - (E ** 2 - px ** 2 - py ** 2 - pz ** 2)
    assert expr1.args == (-E ** 2, px ** 2, py ** 2, pz ** 2, A(i0) * A(-i0))
    expr2 = E ** 2 - px ** 2 - py ** 2 - pz ** 2 - A(i0) * A(-i0)
    assert expr2.args == (E ** 2, -px ** 2, -py ** 2, -pz ** 2, -A(i0) * A(-i0))
    expr3 = A(i0) * A(-i0) - E ** 2 + px ** 2 + py ** 2 + pz ** 2
    assert expr3.args == (-E ** 2, px ** 2, py ** 2, pz ** 2, A(i0) * A(-i0))
    expr4 = B(i1) * B(-i1) + 2 * E ** 2 - 2 * px ** 2 - 2 * py ** 2 - 2 * pz ** 2 - A(i0) * A(-i0)
    assert expr4.args == (2 * E ** 2, -2 * px ** 2, -2 * py ** 2, -2 * pz ** 2, B(i1) * B(-i1), -A(i0) * A(-i0))