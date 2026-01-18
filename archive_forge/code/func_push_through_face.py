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
def push_through_face(self, tri, tet0):

    def can_straighten(arc, tri):
        arc_pts = arc.to_3d_points()
        tri_pts = [p.to_3d_point() for p in tri]
        return pl_utils.can_straighten_bend(arc_pts, tri_pts)
    arc0_a, arc0_b = tri
    a, b, c = (arc0_a.start, arc0_a.end, arc0_b.end)
    assert b == arc0_b.start
    other_arcs = [arc for arc in tet0.arcs if arc != arc0_a and arc != arc0_b]
    assert all((can_straighten(arc, [a, b, c]) for arc in other_arcs))
    back = arc0_a.past.past
    u, v = (back.start, back.end)
    next = arc0_b.next.next
    x, y = (next.start, next.end)
    tet1 = back.tet
    assert back.tet == next.tet
    assert v.on_boundary() and x.on_boundary()
    assert v.zero_coordinates() == x.zero_coordinates()
    uv = BarycentricArc(u, v, tet=tet1)
    xy = BarycentricArc(x, y, tet=tet1)
    other_arcs = [arc for arc in tet1.arcs if arc not in [uv, xy]]
    success = False
    for l in range(1, 12):
        t = QQ(2) ** (-l)
        w = v.convex_combination(u, t)
        z = x.convex_combination(y, t)
        if all((can_straighten(arc, [w, v, x]) for arc in other_arcs)):
            if all((can_straighten(arc, [w, x, z]) for arc in other_arcs)):
                success = True
                break
    if not success:
        return False
    uw = BarycentricArc(u, w, tet=tet1)
    wz = BarycentricArc(w, z, tet=tet1)
    zy = BarycentricArc(z, y, tet=tet1)
    back.past.glue_to(uw)
    uw.glue_to(wz)
    wz.glue_to(zy)
    zy.glue_to(next.next)
    tet0.arcs.remove(arc0_a)
    tet0.arcs.remove(arc0_b)
    tet1.arcs.remove(uv)
    tet1.arcs.remove(xy)
    tet1.arcs += [uw, wz, zy]
    return True