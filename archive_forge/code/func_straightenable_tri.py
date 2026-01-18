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
def straightenable_tri(arcs):
    for arc_a in arcs:
        if not arc_a.end.on_boundary():
            arc_b = arc_a.next
            if arc_a.start.on_boundary() and arc_b.end.on_boundary():
                face_a = arc_a.start.boundary_face()
                face_b = arc_b.end.boundary_face()
                if face_a == face_b:
                    continue
            if can_straighten_bend(arc_a, arc_b, arcs):
                return (arc_a, arc_b)