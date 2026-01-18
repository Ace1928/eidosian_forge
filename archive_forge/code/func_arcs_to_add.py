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
def arcs_to_add(arcs):
    ans = []
    on_boundary = set()
    for arc in arcs:
        trimmed = arc.trim()
        if trimmed:
            a, b = (trimmed.start, trimmed.end)
            zeros_a, zeros_b = (a.zero_coordinates(), b.zero_coordinates())
            if len(zeros_a) > 1 or len(zeros_b) > 1:
                raise GeneralPositionError('Nongeneric interesection')
            if a != b:
                if len(zeros_a) == 1 and zeros_a == zeros_b:
                    raise GeneralPositionError('Lies in face')
                if len(zeros_a) == 1:
                    if a in on_boundary:
                        raise GeneralPositionError('Bounce')
                    on_boundary.add(a)
                if len(zeros_b) == 1:
                    if b in on_boundary:
                        raise GeneralPositionError('Bounce')
                    on_boundary.add(b)
                ans.append(trimmed)
    return ans