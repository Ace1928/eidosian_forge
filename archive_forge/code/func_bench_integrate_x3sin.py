from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
def bench_integrate_x3sin():
    integrate(x ** 3 * sin(x), x)