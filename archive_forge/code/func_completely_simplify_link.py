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
def completely_simplify_link(self, straighten=True, push=True, around_edges=True, limit=10):
    any_success, success, l = (False, True, 0)
    while success and l < limit:
        success = False
        for tet in self:
            if self.simplify_link(tet, straighten, push):
                success, any_success = (True, True)
        l += 1
    return any_success