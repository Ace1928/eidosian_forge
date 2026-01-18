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
def smash_all_edges(self):
    """
        Collapse edges to reduce the number of vertices as much as
        possible. Returns whether the number of vertices has been
        reduced to one.
        """
    success = True
    while len(self.Vertices) > 1 and success:
        success = False
        edges = sorted(self.Edges, key=lambda E: E.valence(), reverse=True)
        edges = self.Edges
        for edge in edges:
            if self.smash_star(edge):
                success = True
                break
    return len(self.Vertices) == 1