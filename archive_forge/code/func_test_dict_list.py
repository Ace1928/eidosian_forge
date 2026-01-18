from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
def test_dict_list():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')
    E = ReferenceFrame('E')
    F = ReferenceFrame('F')
    B.orient_axis(A, A.x, 1.0)
    C.orient_axis(B, B.x, 1.0)
    D.orient_axis(C, C.x, 1.0)
    assert D._dict_list(A, 0) == [D, C, B, A]
    E.orient_axis(D, D.x, 1.0)
    assert C._dict_list(A, 0) == [C, B, A]
    assert C._dict_list(E, 0) == [C, D, E]
    raises(ValueError, lambda: C._dict_list(E, 5))
    raises(ValueError, lambda: F._dict_list(A, 0))