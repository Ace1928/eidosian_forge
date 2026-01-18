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
def test_TensorIndexType():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', metric_name='g', metric_symmetry=1, dim=D, dummy_name='L')
    m0, m1, m2, m3, m4 = tensor_indices('m0:5', Lorentz)
    sym2 = TensorSymmetry.fully_symmetric(2)
    sym2n = TensorSymmetry(*get_symmetric_group_sgs(2))
    assert sym2 == sym2n
    g = Lorentz.metric
    assert str(g) == 'g(Lorentz,Lorentz)'
    assert Lorentz.eps_dim == Lorentz.dim
    TSpace = TensorIndexType('TSpace', dummy_name='TSpace')
    i0, i1 = tensor_indices('i0 i1', TSpace)
    g = TSpace.metric
    A = TensorHead('A', [TSpace] * 2, sym2)
    assert str(A(i0, -i0).canon_bp()) == 'A(TSpace_0, -TSpace_0)'