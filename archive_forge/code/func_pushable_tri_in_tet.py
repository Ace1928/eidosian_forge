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
def pushable_tri_in_tet(arcs):
    for arc_a in arcs:
        if arc_a.start.on_boundary() and arc_a.end.is_interior():
            arc_b = arc_a.next
            start_zeros = arc_a.start.zero_coordinates()
            end_zeros = arc_b.end.zero_coordinates()
            if start_zeros == end_zeros:
                if can_straighten_bend(arc_a, arc_b, arcs):
                    return (arc_a, arc_b)