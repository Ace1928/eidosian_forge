from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def orientnew(self, newname, rot_type, amounts, rot_order='', variables=None, indices=None, latexs=None):
    """Returns a new reference frame oriented with respect to this
        reference frame.

        See ``ReferenceFrame.orient()`` for detailed examples of how to orient
        reference frames.

        Parameters
        ==========

        newname : str
            Name for the new reference frame.
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
        indices : tuple of str
            Enables the reference frame's basis unit vectors to be accessed by
            Python's square bracket indexing notation using the provided three
            indice strings and alters the printing of the unit vectors to
            reflect this choice.
        latexs : tuple of str
            Alters the LaTeX printing of the reference frame's basis unit
            vectors to the provided three valid LaTeX strings.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, vlatex
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = ReferenceFrame('N')

        Create a new reference frame A rotated relative to N through a simple
        rotation.

        >>> A = N.orientnew('A', 'Axis', (q0, N.x))

        Create a new reference frame B rotated relative to N through body-fixed
        rotations.

        >>> B = N.orientnew('B', 'Body', (q1, q2, q3), '123')

        Create a new reference frame C rotated relative to N through a simple
        rotation with unique indices and LaTeX printing.

        >>> C = N.orientnew('C', 'Axis', (q0, N.x), indices=('1', '2', '3'),
        ... latexs=(r'\\hat{\\mathbf{c}}_1',r'\\hat{\\mathbf{c}}_2',
        ... r'\\hat{\\mathbf{c}}_3'))
        >>> C['1']
        C['1']
        >>> print(vlatex(C['1']))
        \\hat{\\mathbf{c}}_1

        """
    newframe = self.__class__(newname, variables=variables, indices=indices, latexs=latexs)
    approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
    rot_order = translate(str(rot_order), 'XYZxyz', '123123')
    rot_type = rot_type.upper()
    if rot_order not in approved_orders:
        raise TypeError('The supplied order is not an approved type')
    if rot_type == 'AXIS':
        newframe.orient_axis(self, amounts[1], amounts[0])
    elif rot_type == 'DCM':
        newframe.orient_explicit(self, amounts)
    elif rot_type == 'BODY':
        newframe.orient_body_fixed(self, amounts, rot_order)
    elif rot_type == 'SPACE':
        newframe.orient_space_fixed(self, amounts, rot_order)
    elif rot_type == 'QUATERNION':
        newframe.orient_quaternion(self, amounts)
    else:
        raise NotImplementedError('That is not an implemented rotation')
    return newframe