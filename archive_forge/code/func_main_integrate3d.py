from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    """Function to translate the problem of integrating uni/bi/tri-variate
    polynomials over a 3-Polytope to integrating over its faces.
    This is done using Generalized Stokes' Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        Max degree of constituent monomial in given list of polynomial.

    Examples
    ========

    >>> from sympy.integrals.intpoly import main_integrate3d,     hyperplane_parameters
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),                (5, 0, 5), (5, 5, 0), (5, 5, 5)],                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> vertices = cube[0]
    >>> faces = cube[1:]
    >>> hp_params = hyperplane_parameters(faces, vertices)
    >>> main_integrate3d(1, faces, vertices, hp_params)
    -125
    """
    result = {}
    dims = (x, y, z)
    dim_length = len(dims)
    if max_degree:
        grad_terms = gradient_terms(max_degree, 3)
        flat_list = [term for z_terms in grad_terms for x_term in z_terms for term in x_term]
        for term in flat_list:
            result[term[0]] = 0
        for facet_count, hp in enumerate(hp_params):
            a, b = (hp[0], hp[1])
            x0 = vertices[facets[facet_count][0]]
            for i, monom in enumerate(flat_list):
                expr, x_d, y_d, z_d, z_index, y_index, x_index, _ = monom
                degree = x_d + y_d + z_d
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = integration_reduction_dynamic(facets, facet_count, a, b, expr, degree, dims, x_index, y_index, z_index, x0, grad_terms, i, vertices, hp)
                monom[7] = value_over_face
                result[expr] += value_over_face * (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        return result
    else:
        integral_value = S.Zero
        polynomials = decompose(expr)
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            for i, facet in enumerate(facets):
                hp = hp_params[i]
                if hp[1].is_zero:
                    continue
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                poly_contribute += pi * (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            poly_contribute /= dim_length + deg
            integral_value += poly_contribute
    return integral_value