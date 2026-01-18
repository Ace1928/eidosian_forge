from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core
def singular_points(self):
    """
        Returns a set of singular points of the region.

        The singular points are those points on the region
        where all partial derivatives vanish.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
        >>> I.singular_points()
        {(1, 1)}

        """
    eq_list = [self.equation]
    for var in self.variables:
        eq_list += [diff(self.equation, var)]
    return nonlinsolve(eq_list, list(self.variables))