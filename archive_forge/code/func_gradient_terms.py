from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def gradient_terms(binomial_power=0, no_of_gens=2):
    """Returns a list of all the possible monomials between
    0 and y**binomial_power for 2D case and z**binomial_power
    for 3D case.

    Parameters
    ==========

    binomial_power :
        Power upto which terms are generated.
    no_of_gens :
        Denotes whether terms are being generated for 2D or 3D case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import gradient_terms
    >>> gradient_terms(2)
    [[1, 0, 0, 0], [y, 0, 1, 0], [y**2, 0, 2, 0], [x, 1, 0, 0],
    [x*y, 1, 1, 0], [x**2, 2, 0, 0]]
    >>> gradient_terms(2, 3)
    [[[[1, 0, 0, 0, 0, 0, 0, 0]]], [[[y, 0, 1, 0, 1, 0, 0, 0],
    [z, 0, 0, 1, 1, 0, 1, 0]], [[x, 1, 0, 0, 1, 1, 0, 0]]],
    [[[y**2, 0, 2, 0, 2, 0, 0, 0], [y*z, 0, 1, 1, 2, 0, 1, 0],
    [z**2, 0, 0, 2, 2, 0, 2, 0]], [[x*y, 1, 1, 0, 2, 1, 0, 0],
    [x*z, 1, 0, 1, 2, 1, 1, 0]], [[x**2, 2, 0, 0, 2, 2, 0, 0]]]]
    """
    if no_of_gens == 2:
        count = 0
        terms = [None] * int((binomial_power ** 2 + 3 * binomial_power + 2) / 2)
        for x_count in range(0, binomial_power + 1):
            for y_count in range(0, binomial_power - x_count + 1):
                terms[count] = [x ** x_count * y ** y_count, x_count, y_count, 0]
                count += 1
    else:
        terms = [[[[x ** x_count * y ** y_count * z ** (z_count - y_count - x_count), x_count, y_count, z_count - y_count - x_count, z_count, x_count, z_count - y_count - x_count, 0] for y_count in range(z_count - x_count, -1, -1)] for x_count in range(0, z_count + 1)] for z_count in range(0, binomial_power + 1)]
    return terms