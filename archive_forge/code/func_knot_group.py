from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def knot_group(self):
    """
        Computes the knot group using the Wirtinger presentation.

        Returns a finitely presented group::

           sage: K = Link('3_1')
           sage: G = K.knot_group()
           sage: type(G)
           <class 'sage.groups.finitely_presented.FinitelyPresentedGroup_with_category'>
        """
    n = len(self.crossings)
    F = FreeGroup(n)
    rels = []
    pieces = self._pieces()
    for z in self.crossings:
        for m, p in enumerate(pieces):
            for t, q in enumerate(p):
                if q[0] == z:
                    if t == 0:
                        j = m
                    elif t == len(p) - 1:
                        i = m
                    else:
                        k = m
        i += 1
        j += 1
        k += 1
        if z.sign > 0:
            r = F([-k, i, k, -j])
        if z.sign < 0:
            r = F([k, i, -k, -j])
        rels.append(r)
    return F / rels