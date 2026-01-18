from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def compare3d(a, b):
    det = cross_product(center, a, b)
    dot_product = sum([det[i] * normal[i] for i in range(0, 3)])
    if dot_product < 0:
        return -order
    elif dot_product > 0:
        return order