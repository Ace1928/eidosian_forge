from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit
@cacheit
def position_wrt(self, other):
    """
        Returns the position vector of this Point with respect to
        another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this Point wrt it is
            returned. If its an instance of CoordSyRect, the position
            wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> N.origin.position_wrt(p1)
        (-10)*N.i

        """
    if not isinstance(other, Point) and (not isinstance(other, CoordSys3D)):
        raise TypeError(str(other) + 'is not a Point or CoordSys3D')
    if isinstance(other, CoordSys3D):
        other = other.origin
    if other == self:
        return Vector.zero
    elif other == self._parent:
        return self._pos
    elif other._parent == self:
        return -1 * other._pos
    rootindex, path = _path(self, other)
    result = Vector.zero
    i = -1
    for i in range(rootindex):
        result += path[i]._pos
    i += 2
    while i < len(path):
        result -= path[i]._pos
        i += 1
    return result