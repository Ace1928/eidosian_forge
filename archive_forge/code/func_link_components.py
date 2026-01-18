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
def link_components(self):
    arcs = sum([tet.arcs for tet in self], [])
    num_arcs = len(arcs)
    components = []
    while len(arcs) > 0:
        arc0 = arcs.pop()
        component = [arc0]
        arc = arc0
        while arc.next != arc0:
            arc = arc.next
            if not isinstance(arc, InfinitesimalArc):
                arcs.remove(arc)
            component.append(arc)
        components.append(component)
    return components