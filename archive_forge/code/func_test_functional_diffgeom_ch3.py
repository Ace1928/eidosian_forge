from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
from sympy.core import symbols, Function, Derivative
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin, cos
from sympy.matrices import Matrix
def test_functional_diffgeom_ch3():
    x0, y0 = symbols('x0, y0', real=True)
    x, y, t = symbols('x, y, t', real=True)
    f = Function('f')
    b1 = Function('b1')
    b2 = Function('b2')
    p_r = R2_r.point([x0, y0])
    s_field = f(R2.x, R2.y)
    v_field = b1(R2.x) * R2.e_x + b2(R2.y) * R2.e_y
    assert v_field.rcall(s_field).rcall(p_r).doit() == b1(x0) * Derivative(f(x0, y0), x0) + b2(y0) * Derivative(f(x0, y0), y0)
    assert R2.e_x(R2.r ** 2).rcall(p_r) == 2 * x0
    v = R2.e_x + 2 * R2.e_y
    s = R2.r ** 2 + 3 * R2.x
    assert v.rcall(s).rcall(p_r).doit() == 2 * x0 + 4 * y0 + 3
    circ = -R2.y * R2.e_x + R2.x * R2.e_y
    series = intcurve_series(circ, t, R2_r.point([1, 0]), coeffs=True)
    series_x, series_y = zip(*series)
    assert all([term == cos(t).taylor_term(i, t) for i, term in enumerate(series_x)])
    assert all([term == sin(t).taylor_term(i, t) for i, term in enumerate(series_y)])