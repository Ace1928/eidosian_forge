from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def find_edge_linking_annuli(self, manifold):
    """
        Surface.find_edge_linking_annuli(mcomplex) returns a list of the
        indices of those edges for which the Surface contains an edge-linking
        annulus (and hence has an obvious compression).
        """
    if self not in manifold.NormalSurfaces:
        raise ValueError('That manifold does not contain the Surface!')
    linked_edges = []
    for edge in manifold.Edges:
        is_linked = 1
        for corner in edge.Corners:
            quad = DisjointQuad[corner.Subsimplex]
            if self.Coefficients[corner.Tetrahedron.Index] == 0 or self.Quadtypes[corner.Tetrahedron.Index] != quad:
                is_linked = 0
                break
        if is_linked:
            linked_edges.append(edge.Index)
    return linked_edges