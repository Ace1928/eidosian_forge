from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def orient(self, parent, rot_type, amounts, rot_order=''):
    """Sets the orientation of this reference frame relative to another
        (parent) reference frame.

        .. note:: It is now recommended to use the ``.orient_axis,
           .orient_body_fixed, .orient_space_fixed, .orient_quaternion``
           methods for the different rotation types.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts :
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3,3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        """
    _check_frame(parent)
    approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
    rot_order = translate(str(rot_order), 'XYZxyz', '123123')
    rot_type = rot_type.upper()
    if rot_order not in approved_orders:
        raise TypeError('The supplied order is not an approved type')
    if rot_type == 'AXIS':
        self.orient_axis(parent, amounts[1], amounts[0])
    elif rot_type == 'DCM':
        self.orient_explicit(parent, amounts)
    elif rot_type == 'BODY':
        self.orient_body_fixed(parent, amounts, rot_order)
    elif rot_type == 'SPACE':
        self.orient_space_fixed(parent, amounts, rot_order)
    elif rot_type == 'QUATERNION':
        self.orient_quaternion(parent, amounts)
    else:
        raise NotImplementedError('That is not an implemented rotation')