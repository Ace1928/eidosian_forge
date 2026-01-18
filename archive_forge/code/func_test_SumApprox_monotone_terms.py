import math
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.codegen.rewriting import optimize
from sympy.codegen.approximations import SumApprox, SeriesApprox
def test_SumApprox_monotone_terms():
    x, y, z = symbols('x y z')
    expr1 = exp(z) * (x ** 2 + y ** 2 + 1)
    bnds1 = {x: (0, 0.001), y: (100, 1000)}
    sum_approx_m2 = SumApprox(bounds=bnds1, reltol=0.01)
    sum_approx_m5 = SumApprox(bounds=bnds1, reltol=1e-05)
    sum_approx_m11 = SumApprox(bounds=bnds1, reltol=1e-11)
    assert (optimize(expr1, [sum_approx_m2]) / exp(z) - y ** 2).simplify() == 0
    assert (optimize(expr1, [sum_approx_m5]) / exp(z) - (y ** 2 + 1)).simplify() == 0
    assert (optimize(expr1, [sum_approx_m11]) / exp(z) - (y ** 2 + 1 + x ** 2)).simplify() == 0