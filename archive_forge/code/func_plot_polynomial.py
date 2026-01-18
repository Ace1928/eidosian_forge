from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def plot_polynomial(expr):
    """Plots the polynomial using the functions written in
    plotting module which in turn uses matplotlib backend.

    Parameter
    =========

    expr:
        Denotes a polynomial(SymPy expression).
    """
    from sympy.plotting.plot import plot3d, plot
    gens = expr.free_symbols
    if len(gens) == 2:
        plot3d(expr)
    else:
        plot(expr)