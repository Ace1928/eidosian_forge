import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def alexander_polynomial(self):
    D = DrorDatum(self.link, self.crossings)
    gluings = self.gluings[:]
    for C, gluings in list(zip(self.crossings, gluings))[:-1]:
        D.add_crossing(C)
        for a, b in gluings:
            D.merge(a, b)
    C = self.crossings[-1]
    D.add_crossing(C)
    for a, b in self.gluings[-1][:-1]:
        D.merge(a, b)
    alex = D.omega
    p, q = (alex.numerator(), alex.denominator())
    assert [abs(c) for c in q.coefficients()] == [1]
    if p.leading_coefficient() < 0:
        p = -p
    t, e = (p.parent().gen(), min(p.exponents()))
    return p // t ** e