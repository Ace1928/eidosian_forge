from sympy.core import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, symbols
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point, Point2D
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D
from sympy.geometry.ellipse import Ellipse
from sympy.functions import sign
from sympy.simplify import simplify
from sympy.solvers.solvers import solve
@property
def p_parameter(self):
    """P is a parameter of parabola.

        Returns
        =======

        p : number or symbolic expression

        Notes
        =====

        The absolute value of p is the focal length. The sign on p tells
        which way the parabola faces. Vertical parabolas that open up
        and horizontal that open right, give a positive value for p.
        Vertical parabolas that open down and horizontal that open left,
        give a negative value for p.


        See Also
        ========

        https://www.sparknotes.com/math/precalc/conicsections/section2/

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.p_parameter
        -4

        """
    m = self.directrix.slope
    if m is S.Infinity:
        x = self.directrix.coefficients[2]
        p = sign(self.focus.args[0] + x)
    elif m == 0:
        y = self.directrix.coefficients[2]
        p = sign(self.focus.args[1] + y)
    else:
        d = self.directrix.projection(self.focus)
        p = sign(self.focus.x - d.x)
    return p * self.focal_length