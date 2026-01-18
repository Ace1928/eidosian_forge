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
def suspension_of_polygon(self, num_sides_of_polygon):
    """
        This method adds the suspension of a triangulation of a
        polygon to self.Tetrahedra and returns::

          (top_arrows, bottom_arrows)

        Currently the choice of triangulation of the polygon is one
        that is the cone over an edge.  Probably this should be
        generalized.  top_arrows and bottom arrows are for gluing in
        this complex via the method ``replace_star``.
        """
    top_tets = self.new_tets(num_sides_of_polygon - 2)
    bottom_tets = self.new_tets(num_sides_of_polygon - 2)
    n = len(top_tets)
    for i in range(n):
        top_tets[i].attach(F3, bottom_tets[i], (0, 2, 1, 3))
    for i in range(n - 1):
        top_tets[i].attach(F0, top_tets[i + 1], (1, 0, 2, 3))
        bottom_tets[i].attach(F0, bottom_tets[i + 1], (2, 1, 0, 3))
    top_arrows = [Arrow(comp(E13), F1, top_tets[0])]
    bottom_arrows = [Arrow(comp(E23), F2, bottom_tets[0])]
    for i in range(n):
        top_arrows.append(Arrow(comp(E23), F2, top_tets[i]))
        bottom_arrows.append(Arrow(comp(E13), F1, bottom_tets[i]))
    top_arrows.append(Arrow(comp(E03), F0, top_tets[i]))
    bottom_arrows.append(Arrow(comp(E03), F0, bottom_tets[i]))
    return (top_arrows, bottom_arrows)