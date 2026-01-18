from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
def is_face_to_vertex(self) -> bool:
    """
        True if line segment starts on a face and goes to a vertex.
        """
    return self.endpoints[0].subsimplex in simplex.TwoSubsimplices and self.endpoints[1].subsimplex in simplex.ZeroSubsimplices