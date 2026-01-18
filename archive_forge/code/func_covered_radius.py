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
def covered_radius(self):
    """
        The pieces in GeodesicTube.pieces cover a tube of radius at least
        the value returned by this function.

        Note that, for convenience, an interval is returned even though
        only the left value is relevant.
        """
    return self._pending_pieces[0].lower_bound