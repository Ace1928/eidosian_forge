from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
from sympy.functions.elementary.miscellaneous import sqrt
def test_modgcd_bivariate_integers():
    R, x, y = ring('x,y', ZZ)
    f, g = (R.zero, R.zero)
    assert modgcd_bivariate(f, g) == (0, 0, 0)
    f, g = (2 * x, R(2))
    assert modgcd_bivariate(f, g) == (2, x, 1)
    f, g = (x + 2 * y, x + y)
    assert modgcd_bivariate(f, g) == (1, f, g)
    f, g = (x ** 2 + 2 * x * y + y ** 2, x ** 3 + y ** 3)
    assert modgcd_bivariate(f, g) == (x + y, x + y, x ** 2 - x * y + y ** 2)
    f, g = (x * y ** 2 + 2 * x * y + x, x * y ** 3 + x)
    assert modgcd_bivariate(f, g) == (x * y + x, y + 1, y ** 2 - y + 1)
    f, g = (x ** 2 * y ** 2 + x ** 2 * y + 1, x * y ** 2 + x * y + 1)
    assert modgcd_bivariate(f, g) == (1, f, g)
    f = 2 * x * y ** 2 + 4 * x * y + 2 * x + y ** 2 + 2 * y + 1
    g = 2 * x * y ** 3 + 2 * x + y ** 3 + 1
    assert modgcd_bivariate(f, g) == (2 * x * y + 2 * x + y + 1, y + 1, y ** 2 - y + 1)
    f, g = (2 * x ** 2 + 4 * x + 2, x + 1)
    assert modgcd_bivariate(f, g) == (x + 1, 2 * x + 2, 1)
    f, g = (x + 1, 2 * x ** 2 + 4 * x + 2)
    assert modgcd_bivariate(f, g) == (x + 1, 1, 2 * x + 2)
    f = 2 * x ** 2 + 4 * x * y - 2 * x - 4 * y
    g = x ** 2 + x - 2
    assert modgcd_bivariate(f, g) == (x - 1, 2 * x + 4 * y, x + 2)
    f = 2 * x ** 2 + 2 * x * y - 3 * x - 3 * y
    g = 4 * x * y - 2 * x + 4 * y ** 2 - 2 * y
    assert modgcd_bivariate(f, g) == (x + y, 2 * x - 3, 4 * y - 2)