from .arrow import Arrow
from .simplex import PickAnEdge
def bdry_arrow(self):
    if self.IntOrBdry != 'bdry':
        return None
    face = self.Corners[0].Subsimplex
    tet = self.Corners[0].Tetrahedron
    edge = PickAnEdge[face]
    return Arrow(edge, face, tet)