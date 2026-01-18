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
def build_edge_classes(self):
    """
        Construct the edge classes and compute valences.
        """
    for tet in self.Tetrahedra:
        for one_subsimplex in OneSubsimplices:
            if tet.Class[one_subsimplex] is None:
                newEdge = Edge()
                self.Edges.append(newEdge)
                first_arrow = Arrow(one_subsimplex, RightFace[one_subsimplex], tet)
                a = first_arrow.copy()
                sanity_check = 0
                boundary_hits = 0
                while 1:
                    if sanity_check > 6 * len(self.Tetrahedra):
                        raise Insanity('Bad gluing data: could not construct edge link.')
                    newEdge._add_corner(a)
                    a.Tetrahedron.Class[a.Edge] = newEdge
                    if a.next() is None:
                        if not boundary_hits == 0:
                            newEdge.RightBdryArrow = a.copy()
                            newEdge.Corners.reverse()
                            break
                        else:
                            boundary_hits = 1
                            newEdge.LeftBdryArrow = a.copy()
                            newEdge.IntOrBdry = 'bdry'
                            a = first_arrow.copy()
                            a.reverse()
                            del newEdge.Corners[0]
                            newEdge.Corners.reverse()
                    elif a == first_arrow:
                        newEdge.IntOrBdry = 'int'
                        break
                    sanity_check = sanity_check + 1
    self.EdgeValences = [edge.valence() for edge in self.Edges]
    for i in range(len(self.Edges)):
        self.Edges[i].Index = i