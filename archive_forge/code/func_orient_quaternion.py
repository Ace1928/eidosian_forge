from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def orient_quaternion(self, parent, numbers):
    """Sets the orientation of this reference frame relative to a parent
        reference frame via an orientation quaternion. An orientation
        quaternion is defined as a finite rotation a unit vector, ``(lambda_x,
        lambda_y, lambda_z)``, by an angle ``theta``. The orientation
        quaternion is described by four parameters:

        - ``q0 = cos(theta/2)``
        - ``q1 = lambda_x*sin(theta/2)``
        - ``q2 = lambda_y*sin(theta/2)``
        - ``q3 = lambda_z*sin(theta/2)``

        See `Quaternions and Spatial Rotation
        <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation>`_ on
        Wikipedia for more information.

        Parameters
        ==========
        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        numbers : 4-tuple of sympifiable
            The four quaternion scalar numbers as defined above: ``q0``,
            ``q1``, ``q2``, ``q3``.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')

        Set the orientation:

        >>> B.orient_quaternion(N, (q0, q1, q2, q3))
        >>> B.dcm(N)
        Matrix([
        [q0**2 + q1**2 - q2**2 - q3**2,             2*q0*q3 + 2*q1*q2,            -2*q0*q2 + 2*q1*q3],
        [           -2*q0*q3 + 2*q1*q2, q0**2 - q1**2 + q2**2 - q3**2,             2*q0*q1 + 2*q2*q3],
        [            2*q0*q2 + 2*q1*q3,            -2*q0*q1 + 2*q2*q3, q0**2 - q1**2 - q2**2 + q3**2]])

        """
    from sympy.physics.vector.functions import dynamicsymbols
    _check_frame(parent)
    numbers = list(numbers)
    for i, v in enumerate(numbers):
        if not isinstance(v, Vector):
            numbers[i] = sympify(v)
    if not isinstance(numbers, (list, tuple)) & (len(numbers) == 4):
        raise TypeError('Amounts are a list or tuple of length 4')
    q0, q1, q2, q3 = numbers
    parent_orient_quaternion = Matrix([[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)], [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)], [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]])
    self._dcm(parent, parent_orient_quaternion)
    t = dynamicsymbols._t
    q0, q1, q2, q3 = numbers
    q0d = diff(q0, t)
    q1d = diff(q1, t)
    q2d = diff(q2, t)
    q3d = diff(q3, t)
    w1 = 2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1)
    w2 = 2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2)
    w3 = 2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3)
    wvec = Vector([(Matrix([w1, w2, w3]), self)])
    self._ang_vel_dict.update({parent: wvec})
    parent._ang_vel_dict.update({self: -wvec})
    self._var_dict = {}