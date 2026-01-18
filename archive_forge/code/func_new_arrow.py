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
def new_arrow(self):
    """
        Add one new tetrahedron and return one of its arrows.
        """
    tet = Tetrahedron()
    self.add_tet(tet)
    return Arrow(E01, F3, tet)