from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def orient_axis(self, parent, axis, angle):
    """Sets the orientation of this reference frame with respect to a
        parent reference frame by rotating through an angle about an axis fixed
        in the parent reference frame.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        axis : Vector
            Vector fixed in the parent frame about about which this frame is
            rotated. It need not be a unit vector and the rotation follows the
            right hand rule.
        angle : sympifiable
            Angle in radians by which it the frame is to be rotated.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.orient_axis(N, N.x, q1)

        The ``orient_axis()`` method generates a direction cosine matrix and
        its transpose which defines the orientation of B relative to N and vice
        versa. Once orient is called, ``dcm()`` outputs the appropriate
        direction cosine matrix:

        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])
        >>> N.dcm(B)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The following two lines show that the sense of the rotation can be
        defined by negating the vector direction or the angle. Both lines
        produce the same result.

        >>> B.orient_axis(N, -N.x, q1)
        >>> B.orient_axis(N, N.x, -q1)

        """
    from sympy.physics.vector.functions import dynamicsymbols
    _check_frame(parent)
    if not isinstance(axis, Vector) and isinstance(angle, Vector):
        axis, angle = (angle, axis)
    axis = _check_vector(axis)
    theta = sympify(angle)
    if not axis.dt(parent) == 0:
        raise ValueError('Axis cannot be time-varying.')
    unit_axis = axis.express(parent).normalize()
    unit_col = unit_axis.args[0][0]
    parent_orient_axis = (eye(3) - unit_col * unit_col.T) * cos(theta) + Matrix([[0, -unit_col[2], unit_col[1]], [unit_col[2], 0, -unit_col[0]], [-unit_col[1], unit_col[0], 0]]) * sin(theta) + unit_col * unit_col.T
    self._dcm(parent, parent_orient_axis)
    thetad = theta.diff(dynamicsymbols._t)
    wvec = thetad * axis.express(parent).normalize()
    self._ang_vel_dict.update({parent: wvec})
    parent._ang_vel_dict.update({self: -wvec})
    self._var_dict = {}