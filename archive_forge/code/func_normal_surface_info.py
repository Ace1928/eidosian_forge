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
def normal_surface_info(self, out=sys.stdout):
    try:
        for surface in self.NormalSurfaces:
            out.write('-------------------------------------\n\n')
            surface.info(self, out)
            out.write('\n')
    except IOError:
        pass