from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def pair_arcs_across_face(face):
    if face.IntOrBdry == 'bdry':
        return []
    a, b = face.Corners
    a_face = a.Subsimplex
    tet_a = a.Tetrahedron
    tet_b = b.Tetrahedron
    arcs_a = tet_a.arcs
    arcs_b = tet_b.arcs
    perm = tet_a.Gluing[a_face]
    face_arcs_a = []
    face_ind = FaceIndex[a_face]
    for arc in arcs_a:
        zero_start_inds = arc.start.zero_coordinates()
        zero_end_inds = arc.end.zero_coordinates()
        if len(zero_start_inds) == 1:
            face_arcs_a = face_arcs_a + [[arc, zero_start_inds[0], 0]]
        if len(zero_end_inds) == 1:
            face_arcs_a = face_arcs_a + [[arc, zero_end_inds[0], 1]]
        elif len(zero_end_inds) > 1 or len(zero_start_inds) > 1:
            assert False
    pairs = []
    for arc in face_arcs_a:
        arc_a = arc[0]
        i = arc[1]
        j = arc[2]
        x = arc_a.start if j == 0 else arc_a.end
        if i == face_ind:
            x_b = x.permute(perm)
            for arc_b in arcs_b:
                u = arc_b.start
                v = arc_b.end
                if x_b == u:
                    pairs = pairs + [[arc_a, arc_b]]
                elif x_b == v:
                    pairs = pairs + [[arc_b, arc_a]]
    return pairs