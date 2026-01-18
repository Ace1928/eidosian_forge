from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
def normalize_alex_poly(p, t):
    """
    Normalize the sign of the leading coefficient and make all
    exponents positive, then return as an ordinary rather than Laurent
    polynomial.
    """
    if len(t) == 1:
        p = p * t[0] ** (-min(p.exponents()))
        if p.coefficients()[-1] < 0:
            p = -p
        p, e = p.polynomial_construction()
        assert e == 0
        return p
    max_degree = max((sum(x) for x in p.exponents()))
    highest_monomial_exps = [x for x in p.exponents() if sum(x) == max_degree]
    leading_exponents = max(highest_monomial_exps)
    leading_monomial = functools.reduce(lambda x, y: x * y, [t[i] ** leading_exponents[i] for i in range(len(t))])
    l = p.monomial_coefficient(leading_monomial)
    if l < 0:
        p = -p
    for i, ti in enumerate(t):
        min_exp = min((x[i] for x in p.exponents()))
        p = p * ti ** (-min_exp)
    R = p.parent()
    p = R.polynomial_ring()(p)
    return p