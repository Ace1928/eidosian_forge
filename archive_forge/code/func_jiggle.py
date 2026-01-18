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
def jiggle(self):
    """
        Do a random ``four_to_four`` move if one is possible.
        """
    fours = [edge for edge in self.Edges if edge.valence() == 4 and edge.IntOrBdry == 'int']
    if len(fours) == 0:
        return False
    return self.four_to_four(random.choice(fours))