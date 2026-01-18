from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
def bench_integrate_x1sin():
    integrate(x ** 1 * sin(x), x)