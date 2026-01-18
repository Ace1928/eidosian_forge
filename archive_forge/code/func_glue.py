from .simplex import *
from .perm4 import Perm4, inv
def glue(self, other):
    if self.Tetrahedron is None and other.Tetrahedron is None:
        return
    if self.Tetrahedron is None:
        other.reverse().glue(self)
        other.reverse()
        return
    if other.Tetrahedron is None:
        self.Tetrahedron.attach(self.Face, None, (0, 1, 2, 3))
        return
    tet0, face0, edge0 = (self.Tetrahedron, self.Face, self.Edge)
    tet1, face1, edge1 = (other.Tetrahedron, other.Face, other.Edge)
    perm, inv_perm, glued_face = _arrow_gluing_dict[edge0, face0, edge1, face1]
    tet0.Neighbor[face0] = tet1
    tet0.Gluing[face0] = perm
    tet1.Neighbor[glued_face] = tet0
    tet1.Gluing[glued_face] = inv_perm