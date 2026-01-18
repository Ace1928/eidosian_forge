from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises
from sympy.polys.specialpolys import (
from sympy.abc import x, y, z
def test_interpolating_poly():
    x0, x1, x2, x3, y0, y1, y2, y3 = symbols('x:4, y:4')
    assert interpolating_poly(0, x) == 0
    assert interpolating_poly(1, x) == y0
    assert interpolating_poly(2, x) == y0 * (x - x1) / (x0 - x1) + y1 * (x - x0) / (x1 - x0)
    assert interpolating_poly(3, x) == y0 * (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2)) + y1 * (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2)) + y2 * (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    assert interpolating_poly(4, x) == y0 * (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3)) + y1 * (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3)) + y2 * (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3)) + y3 * (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2))
    raises(ValueError, lambda: interpolating_poly(2, x, (x, 2), (1, 3)))
    raises(ValueError, lambda: interpolating_poly(2, x, (x + y, 2), (1, 3)))
    raises(ValueError, lambda: interpolating_poly(2, x + y, (x, 2), (1, 3)))
    raises(ValueError, lambda: interpolating_poly(2, 3, (4, 5), (6, 7)))
    raises(ValueError, lambda: interpolating_poly(2, 3, (4, 5), (6, 7, 8)))
    assert interpolating_poly(0, x, (1, 2), (3, 4)) == 0
    assert interpolating_poly(1, x, (1, 2), (3, 4)) == 3
    assert interpolating_poly(2, x, (1, 2), (3, 4)) == x + 2