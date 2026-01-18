from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def red_groebner(G, ring):
    """
    Compute reduced Groebner basis, from BeckerWeispfenning93, p. 216

    Selects a subset of generators, that already generate the ideal
    and computes a reduced Groebner basis for them.
    """

    def reduction(P):
        """
        The actual reduction algorithm.
        """
        Q = []
        for i, p in enumerate(P):
            h = p.rem(P[:i] + P[i + 1:])
            if h:
                Q.append(h)
        return [p.monic() for p in Q]
    F = G
    H = []
    while F:
        f0 = F.pop()
        if not any((monomial_divides(f.LM, f0.LM) for f in F + H)):
            H.append(f0)
    return reduction(H)