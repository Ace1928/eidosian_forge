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
def clear_tet(self, tet):
    """
        Remove the face, edge and vertex classes of a tetrahedron.
        This should destroy the faces, edges and vertices that meet
        the tetrahedron.  A call to build_face_classes,
        build_edge_classes or build_vertex_classes will then rebuild
        the neighborhood without having to rebuild the whole manifold.
        """
    for two_subsimplex in TwoSubsimplices:
        face = tet.Class[two_subsimplex]
        if face is not None:
            face.erase()
        try:
            self.Faces.remove(face)
        except ValueError:
            pass
    for one_subsimplex in OneSubsimplices:
        edge = tet.Class[one_subsimplex]
        if edge is not None:
            edge.erase()
        try:
            self.Edges.remove(edge)
        except ValueError:
            pass
    for zero_subsimplex in ZeroSubsimplices:
        vertex = tet.Class[zero_subsimplex]
        if vertex is not None:
            vertex.erase()
        try:
            self.Vertices.remove(vertex)
        except ValueError:
            pass