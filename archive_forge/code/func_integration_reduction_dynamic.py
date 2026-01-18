from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def integration_reduction_dynamic(facets, index, a, b, expr, degree, dims, x_index, y_index, max_index, x0, monomial_values, monom_index, vertices=None, hp_param=None):
    """The same integration_reduction function which uses a dynamic
    programming approach to compute terms by using the values of the integral
    of previously computed terms.

    Parameters
    ==========

    facets :
        Facets of the Polytope.
    index :
        Index of facet to find intersections with.(Used in left_integral()).
    a, b :
        Hyperplane parameters.
    expr :
        Input monomial.
    degree :
        Total degree of ``expr``.
    dims :
        Tuple denoting axes variables.
    x_index :
        Exponent of 'x' in ``expr``.
    y_index :
        Exponent of 'y' in ``expr``.
    max_index :
        Maximum exponent of any monomial in ``monomial_values``.
    x0 :
        First point on ``facets[index]``.
    monomial_values :
        List of monomial values constituting the polynomial.
    monom_index :
        Index of monomial whose integration is being found.
    vertices : optional
        Coordinates of vertices constituting the 3-Polytope.
    hp_param : optional
        Hyperplane Parameter of the face of the facets[index].

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import (integration_reduction_dynamic,             hyperplane_parameters)
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> x0 = facets[0].points[0]
    >>> monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],                           [y, 0, 1, 15], [x, 1, 0, None]]
    >>> integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1,                                      x0, monomial_values, 3)
    25/2
    """
    value = S.Zero
    m = len(facets)
    if expr == S.Zero:
        return expr
    if len(dims) == 2:
        if not expr.is_number:
            _, x_degree, y_degree, _ = monomial_values[monom_index]
            x_index = monom_index - max_index + x_index - 2 if x_degree > 0 else 0
            y_index = monom_index - 1 if y_degree > 0 else 0
            x_value, y_value = (monomial_values[x_index][3], monomial_values[y_index][3])
            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1]
        value += left_integral2D(m, index, facets, x0, expr, dims)
    else:
        z_index = max_index
        if not expr.is_number:
            x_degree, y_degree, z_degree = (y_index, z_index - x_index - y_index, x_index)
            x_value = monomial_values[z_index - 1][y_index - 1][x_index][7] if x_degree > 0 else 0
            y_value = monomial_values[z_index - 1][y_index][x_index][7] if y_degree > 0 else 0
            z_value = monomial_values[z_index - 1][y_index][x_index - 1][7] if z_degree > 0 else 0
            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1] + z_degree * z_value * x0[2]
        value += left_integral3D(facets, index, expr, vertices, hp_param, degree)
    return value / (len(dims) + degree - 1)