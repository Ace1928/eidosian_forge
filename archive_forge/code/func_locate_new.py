from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit
def locate_new(self, name, position):
    """
        Returns a new Point located at the given position wrt this
        Point.
        Thus, the position vector of the new Point wrt this one will
        be equal to the given 'position' parameter.

        Parameters
        ==========

        name : str
            Name of the new point

        position : Vector
            The position vector of the new Point wrt this one

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p1.position_wrt(N.origin)
        10*N.i

        """
    return Point(name, position, self)