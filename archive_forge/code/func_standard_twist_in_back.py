from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import VERBOSE
from .mcomplex_with_memory import McomplexWithMemory, edge_and_arrow
def standard_twist_in_back(arrow):
    a = arrow.copy()
    assert a.axis().valence() == 2
    b = a.glued()
    c = b.copy().rotate(1)
    d = c.glued()
    assert d.Tetrahedron == b.Tetrahedron
    return c.tail() == d.tail()