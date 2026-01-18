from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
def series_small_a_small_b():
    """Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5.

    Be aware of cancellation of poles in b=0 of digamma(b)/Gamma(b) and
    polygamma functions.

    digamma(b)/Gamma(b) = -1 - 2*M_EG*b + O(b^2)
    digamma(b)^2/Gamma(b) = 1/b + 3*M_EG + b*(-5/12*PI^2+7/2*M_EG^2) + O(b^2)
    polygamma(1, b)/Gamma(b) = 1/b + M_EG + b*(1/12*PI^2 + 1/2*M_EG^2) + O(b^2)
    and so on.
    """
    order = 5
    a, b, x, k = symbols('a b x k')
    M_PI, M_EG, M_Z3 = symbols('M_PI M_EG M_Z3')
    c_subs = {pi: M_PI, EulerGamma: M_EG, zeta(3): M_Z3}
    A = []
    X = []
    B = []
    C = []
    expression = gamma(b) / sympy.exp(x) * Sum(x ** k / factorial(k) / gamma(a * k + b), (k, 0, S.Infinity))
    for n in range(0, order + 1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        x_part = term.subs(polygamma(0, b), 1).replace(polygamma, lambda *args: 0)
        x_part *= (-1) ** n
        pg_part = term / x_part / gamma(b)
        if n >= 1:
            pg_part = pg_part.replace(polygamma, lambda k, x: pg_series(k, x, order + 1 + n))
            pg_part = pg_part.series(b, 0, n=order + 1 - n).removeO().subs(polygamma(2, 1), -2 * zeta(3)).simplify()
        A.append(a ** n / factorial(n))
        X.append(horner(x_part))
        B.append(pg_part)
    C = sympy.Poly(B[1].subs(c_subs), b).coeffs()
    C.reverse()
    for i in range(len(C)):
        C[i] = (C[i] * factorial(i)).simplify()
    s = 'Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5.'
    s += '\nPhi(a, b, x) = exp(x) * sum(A[i] * X[i] * B[i], i=0..5)\n'
    s += 'B[0] = 1\n'
    s += 'B[i] = sum(C[k+i-1] * b**k/k!, k=0..)\n'
    s += '\nM_PI = pi'
    s += '\nM_EG = EulerGamma'
    s += '\nM_Z3 = zeta(3)'
    for name, c in zip(['A', 'X'], [A, X]):
        for i in range(len(c)):
            s += f'\n{name}[{i}] = '
            s += str(c[i])
    for i in range(len(C)):
        s += f'\n# C[{i}] = '
        s += str(C[i])
        s += f'\nC[{i}] = '
        s += str(C[i].subs({M_EG: EulerGamma, M_PI: pi, M_Z3: zeta(3)}).evalf(17))
    s += '\n\nTest if B[i] does have the assumed structure.'
    s += '\nC[i] are derived from B[1] alone.'
    s += '\nTest B[2] == C[1] + b*C[2] + b^2/2*C[3] + b^3/6*C[4] + ..'
    test = sum([b ** k / factorial(k) * C[k + 1] for k in range(order - 1)])
    test = (test - B[2].subs(c_subs)).simplify()
    s += f'\ntest successful = {test == S(0)}'
    s += '\nTest B[3] == C[2] + b*C[3] + b^2/2*C[4] + ..'
    test = sum([b ** k / factorial(k) * C[k + 2] for k in range(order - 2)])
    test = (test - B[3].subs(c_subs)).simplify()
    s += f'\ntest successful = {test == S(0)}'
    return s