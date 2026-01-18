from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def orient_explicit(self, parent, dcm):
    """Sets the orientation of this reference frame relative to a parent
        reference frame by explicitly setting the direction cosine matrix.

        Parameters
        ==========

        parent : ReferenceFrame
            Reference frame that this reference frame will be rotated relative
            to.
        dcm : Matrix, shape(3, 3)
            Direction cosine matrix that specifies the relative rotation
            between the two reference frames.

        Warns
        ======

        UserWarning
            If the orientation creates a kinematic loop.

        Examples
        ========

        Setup variables for the examples:

        >>> from sympy import symbols, Matrix, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> A = ReferenceFrame('A')
        >>> B = ReferenceFrame('B')
        >>> N = ReferenceFrame('N')

        A simple rotation of ``A`` relative to ``N`` about ``N.x`` is defined
        by the following direction cosine matrix:

        >>> dcm = Matrix([[1, 0, 0],
        ...               [0, cos(q1), -sin(q1)],
        ...               [0, sin(q1), cos(q1)]])
        >>> A.orient_explicit(N, dcm)
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        This is equivalent to using ``orient_axis()``:

        >>> B.orient_axis(N, N.x, q1)
        >>> B.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        **Note carefully that** ``N.dcm(B)`` **(the transpose) would be passed
        into** ``orient_explicit()`` **for** ``A.dcm(N)`` **to match**
        ``B.dcm(N)``:

        >>> A.orient_explicit(N, N.dcm(B))
        >>> A.dcm(N)
        Matrix([
        [1,       0,      0],
        [0,  cos(q1), sin(q1)],
        [0, -sin(q1), cos(q1)]])

        """
    _check_frame(parent)
    if not isinstance(dcm, MatrixBase):
        raise TypeError('Amounts must be a SymPy Matrix type object.')
    parent_orient_dcm = dcm
    self._dcm(parent, parent_orient_dcm)
    wvec = self._w_diff_dcm(parent)
    self._ang_vel_dict.update({parent: wvec})
    parent._ang_vel_dict.update({self: -wvec})
    self._var_dict = {}