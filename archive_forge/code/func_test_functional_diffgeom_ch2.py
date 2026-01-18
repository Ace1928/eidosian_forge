from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
from sympy.core import symbols, Function, Derivative
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin, cos
from sympy.matrices import Matrix
def test_functional_diffgeom_ch2():
    x0, y0, r0, theta0 = symbols('x0, y0, r0, theta0', real=True)
    x, y = symbols('x, y', real=True)
    f = Function('f')
    assert R2_p.point_to_coords(R2_r.point([x0, y0])) == Matrix([sqrt(x0 ** 2 + y0 ** 2), atan2(y0, x0)])
    assert R2_r.point_to_coords(R2_p.point([r0, theta0])) == Matrix([r0 * cos(theta0), r0 * sin(theta0)])
    assert R2_p.jacobian(R2_r, [r0, theta0]) == Matrix([[cos(theta0), -r0 * sin(theta0)], [sin(theta0), r0 * cos(theta0)]])
    field = f(R2.x, R2.y)
    p1_in_rect = R2_r.point([x0, y0])
    p1_in_polar = R2_p.point([sqrt(x0 ** 2 + y0 ** 2), atan2(y0, x0)])
    assert field.rcall(p1_in_rect) == f(x0, y0)
    assert field.rcall(p1_in_polar) == f(x0, y0)
    p_r = R2_r.point([x0, y0])
    p_p = R2_p.point([r0, theta0])
    assert R2.x(p_r) == x0
    assert R2.x(p_p) == r0 * cos(theta0)
    assert R2.r(p_p) == r0
    assert R2.r(p_r) == sqrt(x0 ** 2 + y0 ** 2)
    assert R2.theta(p_r) == atan2(y0, x0)
    h = R2.x * R2.r ** 2 + R2.y ** 3
    assert h.rcall(p_r) == x0 * (x0 ** 2 + y0 ** 2) + y0 ** 3
    assert h.rcall(p_p) == r0 ** 3 * sin(theta0) ** 3 + r0 ** 3 * cos(theta0)