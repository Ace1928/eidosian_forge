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
def straighten_arcs(arcs):
    any_success, success = (False, True)
    offset = 0
    obstructions = dict()
    while success:
        success = False
        for b in arcs[offset:] + arcs[:offset]:
            if not b.end.on_boundary():
                c = b.next
                if b.start.on_boundary() and c.end.on_boundary() and (b.start.boundary_face() == c.end.boundary_face()):
                    continue
                if (b, c) in obstructions:
                    if obstructions[b, c] in arcs:
                        continue
                    else:
                        obstructions.pop((b, c))
                poss, obs = can_straighten_bend(b, c, arcs, True)
                if poss:
                    a, d = (b.past, c.next)
                    if hasattr(b, 'tet'):
                        new_arc = type(b)(b.start, c.end, tet=b.tet)
                    else:
                        new_arc = type(b)(b.start, c.end)
                    if a is not None:
                        a.glue_to(new_arc)
                    if d is not None:
                        new_arc.glue_to(d)
                    arcs.remove(c)
                    offset = arcs.index(b)
                    arcs.remove(b)
                    arcs.append(new_arc)
                    success, any_success = (True, True)
                    break
                else:
                    obstructions[b, c] = obs
    return any_success