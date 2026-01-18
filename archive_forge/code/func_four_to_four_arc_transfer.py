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
def four_to_four_arc_transfer(old_arrow, new_arrows, north_pole=None):
    arcs_in_R3 = []
    for old_tet, emb in barycentric_quad_embedding0(old_arrow, north_pole):
        arcs_in_R3.extend(emb.transfer_arcs_to_R3(old_tet.arcs))
    b = new_arrows[0]
    for new_tet, emb in barycentric_quad_embedding1(b, north_pole):
        new_tet.arcs = arcs_to_add(emb.transfer_arcs_from_R3(arcs_in_R3))