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
def test_canonicalize2():
    D = Symbol('D')
    Eucl = TensorIndexType('Eucl', metric_symmetry=1, dim=D, dummy_name='E')
    i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14 = tensor_indices('i0:15', Eucl)
    A = TensorHead('A', [Eucl] * 3, TensorSymmetry.fully_symmetric(-3))
    t = A(i0, i1, i2) * A(-i1, i3, i4) * A(-i3, i7, i5) * A(-i2, -i5, i6) * A(-i4, -i6, i8)
    t1 = t.canon_bp()
    assert t1 == 0
    t = A(i0, i1, i2) * A(-i1, i3, i4) * A(-i2, i5, i6) * A(-i3, i7, i8) * A(-i6, -i7, i9) * A(-i8, i10, i13) * A(-i5, -i10, i11) * A(-i4, -i11, i12) * A(-i9, -i12, i14)
    t1 = t.canon_bp()
    assert t1 == 0