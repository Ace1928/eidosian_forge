from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.physics.vector import Vector, express
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import _check_vector
def is_conservative(field):
    """
    Checks if a field is conservative.

    Parameters
    ==========

    field : Vector
        The field to check for conservative property

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import is_conservative
    >>> R = ReferenceFrame('R')
    >>> is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)
    True
    >>> is_conservative(R[2] * R.y)
    False

    """
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return curl(field, frame).simplify() == Vector(0)