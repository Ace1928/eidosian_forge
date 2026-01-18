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
def edge_and_arrow(edge_or_arrow):
    """
    Given and edge or an arrow, returns the corresponding compatible
    (edge, arrow) pair.
    """
    if isinstance(edge_or_arrow, Edge):
        edge = edge_or_arrow
        arrow = Arrow(edge.Corners[0].Subsimplex, LeftFace[edge.Corners[0].Subsimplex], edge.Corners[0].Tetrahedron)
    else:
        if not isinstance(edge_or_arrow, Arrow):
            raise ValueError('Input edge_or_arrow is neither')
        arrow = edge_or_arrow.copy()
        edge = arrow.axis()
    return (edge, arrow)