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
def four_to_four(self, edge_or_arrow, must_succeed=False, unsafe_mode=False):
    """
        Replace an edge of valence 4 by another diagonal of the
        octahedron formed by the star of the edge.  There are two
        choices for this diagonal.  If you care which one is used then
        pass an arrow representing the edge of valence four.  The head
        of the arrow will be an endpoint of the new diagonal.  If you
        don't care, just pass an edge.  The choice of diagonal will
        then be made randomly.

        Options and return value are the same as ``two_to_three``.
        """
    edge, a = edge_and_arrow(edge_or_arrow)
    a_orig = a.copy()
    possible, reason = self._edge_permits_four_to_four(edge)
    if not possible:
        if must_succeed:
            raise ValueError(reason)
        return False
    c = self.new_arrows(4)
    c_orig = [x.copy() for x in c]
    for i in range(4):
        c[i].glue(c[(i + 1) % 4])
    b = a.glued().reverse()
    c[0].opposite().glue(a.rotate(1).glued())
    c[1].opposite().glue(b.rotate(-1).glued())
    c[2].opposite().glue(b.rotate(-1).glued())
    c[3].opposite().glue(a.rotate(1).glued())
    a.rotate(1).reverse().next()
    b.rotate(-1).reverse().next()
    c[0].reverse().glue(a.rotate(-1).glued())
    c[1].reverse().glue(b.rotate(1).glued())
    c[2].reverse().glue(b.rotate(1).glued())
    c[3].reverse().glue(a.rotate(-1).glued())
    self._four_to_four_move_hook(a_orig, c_orig)
    for corner in edge.Corners:
        self.delete_tet(corner.Tetrahedron)
    if not unsafe_mode:
        self.build_edge_classes()
        if VERBOSE:
            print('4->4')
            print(self.EdgeValences)
    return True