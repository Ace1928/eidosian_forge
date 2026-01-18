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
def test_canonicalize_no_dummies():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a, b, c, d', Lorentz)
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(c) * A(b) * A(a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a)*A(b)*A(c)'
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    t = A(c) * A(b) * A(a)
    tc = t.canon_bp()
    assert str(tc) == '-A(a)*A(b)*A(c)'
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(b, d) * A(c, a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, c)*A(b, d)'
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2), 1)
    t = A(b, d) * A(c, a)
    tc = t.canon_bp()
    assert str(tc) == '-A(a, c)*A(b, d)'
    t = A(c, a) * A(b, d)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, c)*A(b, d)'