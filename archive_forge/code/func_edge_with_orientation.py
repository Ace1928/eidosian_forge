from collections import OrderedDict
from ... import sage_helper
def edge_with_orientation(self):
    v = self.vertex
    w = [2, 0, 1][v]
    E = self.triangle.edges[v, w]
    S = Side(self.triangle, (v, w))
    return (E, E.orientation_with_respect_to(S))