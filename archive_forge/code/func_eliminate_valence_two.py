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
def eliminate_valence_two(self):
    """
        Perform a single ``two_to_zero`` move on a valence 2 edge, if
        any such is possible.
        """
    did_simplify = False
    progress = True
    while progress:
        progress = False
        for edge in self.Edges:
            if edge.valence() == 2:
                if self.two_to_zero(edge):
                    progress, did_simplify = (True, True)
                    break
    return did_simplify