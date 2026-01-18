from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
def bench_integrate_x2sin():
    integrate(x ** 2 * sin(x), x)