from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
from sympy.core import symbols, Function, Derivative
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin, cos
from sympy.matrices import Matrix
def test_functional_diffgeom_ch4():
    x0, y0, theta0 = symbols('x0, y0, theta0', real=True)
    x, y, r, theta = symbols('x, y, r, theta', real=True)
    r0 = symbols('r0', positive=True)
    f = Function('f')
    b1 = Function('b1')
    b2 = Function('b2')
    p_r = R2_r.point([x0, y0])
    p_p = R2_p.point([r0, theta0])
    f_field = b1(R2.x, R2.y) * R2.dx + b2(R2.x, R2.y) * R2.dy
    assert f_field.rcall(R2.e_x).rcall(p_r) == b1(x0, y0)
    assert f_field.rcall(R2.e_y).rcall(p_r) == b2(x0, y0)
    s_field_r = f(R2.x, R2.y)
    df = Differential(s_field_r)
    assert df(R2.e_x).rcall(p_r).doit() == Derivative(f(x0, y0), x0)
    assert df(R2.e_y).rcall(p_r).doit() == Derivative(f(x0, y0), y0)
    s_field_p = f(R2.r, R2.theta)
    df = Differential(s_field_p)
    assert trigsimp(df(R2.e_x).rcall(p_p).doit()) == cos(theta0) * Derivative(f(r0, theta0), r0) - sin(theta0) * Derivative(f(r0, theta0), theta0) / r0
    assert trigsimp(df(R2.e_y).rcall(p_p).doit()) == sin(theta0) * Derivative(f(r0, theta0), r0) + cos(theta0) * Derivative(f(r0, theta0), theta0) / r0
    assert R2.dx(R2.e_x).rcall(p_r) == 1
    assert R2.dx(R2.e_x) == 1
    assert R2.dx(R2.e_y).rcall(p_r) == 0
    assert R2.dx(R2.e_y) == 0
    circ = -R2.y * R2.e_x + R2.x * R2.e_y
    assert R2.dx(circ).rcall(p_r).doit() == -y0
    assert R2.dy(circ).rcall(p_r) == x0
    assert R2.dr(circ).rcall(p_r) == 0
    assert simplify(R2.dtheta(circ).rcall(p_r)) == 1
    assert (circ - R2.e_theta).rcall(s_field_r).rcall(p_r) == 0