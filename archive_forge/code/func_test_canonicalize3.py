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
def test_canonicalize3():
    D = Symbol('D')
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    a0, a1, a2, a3, a4 = tensor_indices('a0:5', Spinor)
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)
    t = chi(a1) * psi(a0)
    t1 = t.canon_bp()
    assert t1 == t
    t = psi(a1) * chi(a0)
    t1 = t.canon_bp()
    assert t1 == -chi(a0) * psi(a1)