from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def attack_valence_one(self):
    """
        Modify the triangulation near a valence 1 edge, creating a
        valence 2 edge that can likely be eliminated, reducing the
        number of tetrahedra by one.
        """
    if len(self) == 1:
        return False
    for e in self.Edges:
        if e.valence() == 1:
            corner = e.Corners[0]
            tet = corner.Tetrahedron
            sub = corner.Subsimplex
            other_faces = [face for face in TwoSubsimplices if not is_subset(sub, face)]
            assert len(other_faces) == 2
            face = other_faces[0]
            self.two_to_three(face, tet, must_succeed=True)
            return True
    return False