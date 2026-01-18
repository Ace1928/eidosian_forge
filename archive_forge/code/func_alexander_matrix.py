from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def alexander_matrix(self, mv=True):
    """
        Returns the Alexander matrix of the link::

            sage: L = Link('3_1')
            sage: A = L.alexander_matrix()
            sage: A                           # doctest: +SKIP
            ([   -1     t 1 - t]
            [1 - t    -1     t]
            [    t 1 - t    -1], [t, t, t])

            sage: L = Link([(4,1,3,2),(1,4,2,3)])
            sage: A = L.alexander_matrix()
            sage: A                           # doctest: +SKIP
            ([      -1 + t1^-1 t1^-1*t2 - t1^-1]
            [t1*t2^-1 - t2^-1       -1 + t2^-1], [t2, t1])
        """
    comp = len(self.link_components)
    if comp < 2:
        mv = False
    G = self.knot_group()
    num_gens = len(G.gens())
    L_g = LaurentPolynomialRing(QQ, [f'g{i + 1}' for i in range(num_gens)])
    g = list(L_g.gens())
    if mv:
        L_t = LaurentPolynomialRing(QQ, [f't{i + 1}' for i in range(comp)])
        t = list(L_t.gens())
        g_component = [c.strand_components[2] for c in self.crossings]
        for i, gci in enumerate(g_component):
            g[i] = t[gci]
    else:
        L_t = LaurentPolynomialRing(QQ, 't')
        t = L_t.gen()
        g = [t] * len(g)
    B = G.alexander_matrix(g)
    return (B, g)