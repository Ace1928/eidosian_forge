from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def sig_cmp(u, v, order):
    """
    Compare two signatures by extending the term order to K[X]^n.

    u < v iff
        - the index of v is greater than the index of u
    or
        - the index of v is equal to the index of u and u[0] < v[0] w.r.t. order

    u > v otherwise
    """
    if u[1] > v[1]:
        return -1
    if u[1] == v[1]:
        if order(u[0]) < order(v[0]):
            return -1
    return 1