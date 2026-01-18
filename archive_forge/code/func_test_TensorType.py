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
def test_TensorType():
    with warns_deprecated_sympy():
        sym2 = TensorSymmetry.fully_symmetric(2)
        Lorentz = TensorIndexType('Lorentz')
        S2 = TensorType([Lorentz] * 2, sym2)
        assert isinstance(S2, TensorType)