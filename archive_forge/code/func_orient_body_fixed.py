from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def orient_body_fixed(self, parent, angles, rotation_order):
    """Rotates this reference frame relative to the parent reference frame
        by right hand rotating through three successive body fixed simple axis
        rotations. Each subsequent axis of rotation is about the "body fixed"
        unit vectors of a new intermediate reference frame. This type of
        rotation is also referred to rotating through the `Euler and Tait-Bryan
        Angles`_.

        .. _Euler and Tait-Bryan Angles: https://en.wikipedia.org/wiki/Euler_angles

        The computed angular velocity in this method is by default expressed in
        the child's frame, so it is most preferable to use ``u1 * child.x + u2 *
        child.y + u3 * child.z`` as generalized speeds.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations about each intermediate reference frames'
            unit vectors. The Euler rotation about the X, Z', X'' axes can be
            specified by the strings ``'XZX'``, ``'131'``, or the integer
            ``131``. There are 12 unique valid rotation orders (6 Euler and 6
            Tait-Bryan): zxz, xyx, yzy, zyz, xzx, yxy, xyz, yzx, zxy, xzy, zyx,
            and yxz.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1, q2, q3 = symbols('q1, q2, q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B1 = ReferenceFrame('B1')
        >>> B2 = ReferenceFrame('B2')
        >>> B3 = ReferenceFrame('B3')

        For example, a classic Euler Angle rotation can be done by:

        >>> B.orient_body_fixed(N, (q1, q2, q3), 'XYX')
        >>> B.dcm(N)
        Matrix([
        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],
        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],
        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])

        This rotates reference frame B relative to reference frame N through
        ``q1`` about ``N.x``, then rotates B again through ``q2`` about
        ``B.y``, and finally through ``q3`` about ``B.x``. It is equivalent to
        three successive ``orient_axis()`` calls:

        >>> B1.orient_axis(N, N.x, q1)
        >>> B2.orient_axis(B1, B1.y, q2)
        >>> B3.orient_axis(B2, B2.x, q3)
        >>> B3.dcm(N)
        Matrix([
        [        cos(q2),                            sin(q1)*sin(q2),                           -sin(q2)*cos(q1)],
        [sin(q2)*sin(q3), -sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3),  sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)],
        [sin(q2)*cos(q3), -sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1), -sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3)]])

        Acceptable rotation orders are of length 3, expressed in as a string
        ``'XYZ'`` or ``'123'`` or integer ``123``. Rotations about an axis
        twice in a row are prohibited.

        >>> B.orient_body_fixed(N, (q1, q2, 0), 'ZXZ')
        >>> B.orient_body_fixed(N, (q1, q2, 0), '121')
        >>> B.orient_body_fixed(N, (q1, q2, q3), 123)

        """
    from sympy.physics.vector.functions import dynamicsymbols
    _check_frame(parent)
    amounts, rot_order, rot_matrices = self._parse_consecutive_rotations(angles, rotation_order)
    self._dcm(parent, rot_matrices[0] * rot_matrices[1] * rot_matrices[2])
    rot_vecs = [zeros(3, 1) for _ in range(3)]
    for i, order in enumerate(rot_order):
        rot_vecs[i][order - 1] = amounts[i].diff(dynamicsymbols._t)
    u1, u2, u3 = rot_vecs[2] + rot_matrices[2].T * (rot_vecs[1] + rot_matrices[1].T * rot_vecs[0])
    wvec = u1 * self.x + u2 * self.y + u3 * self.z
    self._ang_vel_dict.update({parent: wvec})
    parent._ang_vel_dict.update({self: -wvec})
    self._var_dict = {}