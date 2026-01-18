from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Rational
from sympy.integrals.intpoly import (decompose, best_origin, distance_to_side,
from sympy.geometry.line import Segment2D
from sympy.geometry.polygon import Polygon
from sympy.geometry.point import Point, Point2D
from sympy.abc import x, y, z
from sympy.testing.pytest import slow
def test_main_integrate3d():
    cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0), (5, 0, 5), (5, 5, 0), (5, 5, 5)], [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0], [3, 1, 0, 2], [0, 4, 6, 2]]
    vertices = cube[0]
    faces = cube[1:]
    hp_params = hyperplane_parameters(faces, vertices)
    assert main_integrate3d(1, faces, vertices, hp_params) == -125
    assert main_integrate3d(1, faces, vertices, hp_params, max_degree=1) == {1: -125, y: Rational(-625, 2), z: Rational(-625, 2), x: Rational(-625, 2)}