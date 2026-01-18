from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def lbp_cmp(f, g):
    """
    Compare two labeled polynomials.

    f < g iff
        - Sign(f) < Sign(g)
    or
        - Sign(f) == Sign(g) and Num(f) > Num(g)

    f > g otherwise
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) == -1:
        return -1
    if Sign(f) == Sign(g):
        if Num(f) > Num(g):
            return -1
    return 1