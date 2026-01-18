from . import constants
from . import exceptions
from . import epsilons
from .line import distance_r13_lines, R13Line, R13LineWithMatrix
from .geodesic_info import GeodesicInfo, LiftedTetrahedron
from .quotient_space import balance_end_points_of_line, ZQuotientLiftedTetrahedronSet
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
import heapq
from typing import Sequence, Any
def make_r13_unit_tangent_vector(direction, point):
    s = r13_dot(direction, point)
    return space_r13_normalise(direction + s * point)