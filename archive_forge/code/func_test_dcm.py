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
def test_dcm():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    D = N.orientnew('D', 'Axis', [q4, N.y])
    E = N.orientnew('E', 'Space', [q1, q2, q3], '123')
    assert N.dcm(C) == Matrix([[-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)], [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    test_mat = D.dcm(C) - Matrix([[cos(q1) * cos(q3) * cos(q4) - sin(q3) * (-sin(q4) * cos(q2) + sin(q1) * sin(q2) * cos(q4)), -sin(q2) * sin(q4) - sin(q1) * cos(q2) * cos(q4), sin(q3) * cos(q1) * cos(q4) + cos(q3) * (-sin(q4) * cos(q2) + sin(q1) * sin(q2) * cos(q4))], [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)], [sin(q4) * cos(q1) * cos(q3) - sin(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4)), sin(q2) * cos(q4) - sin(q1) * sin(q4) * cos(q2), sin(q3) * sin(q4) * cos(q1) + cos(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4))]])
    assert test_mat.expand() == zeros(3, 3)
    assert E.dcm(N) == Matrix([[cos(q2) * cos(q3), sin(q3) * cos(q2), -sin(q2)], [sin(q1) * sin(q2) * cos(q3) - sin(q3) * cos(q1), sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), sin(q1) * cos(q2)], [sin(q1) * sin(q3) + sin(q2) * cos(q1) * cos(q3), -sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2)]])