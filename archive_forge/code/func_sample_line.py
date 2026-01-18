from .line import R13LineWithMatrix
from . import epsilons
from . import constants
from . import exceptions
from ..hyperboloid import r13_dot, o13_inverse, distance_unit_time_r13_points # type: ignore
from ..snap.t3mlite import simplex # type: ignore
from ..snap.t3mlite import Tetrahedron, Vertex, Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import matrix # type: ignore
from typing import Tuple, Sequence, Optional, Any
def sample_line(line_with_matrix: R13LineWithMatrix):
    """
    Pick a point on a line in the hyperboloid model.
    Returns an unnormalised time-like vector computed
    as the weighted average of the two light-like
    endpoints of the line.

    The ratio of the weights is some fixed number picked at random so
    that we avoid picking a point that lies, e.g., on an edge of the
    triangulation (which happens for some geodesics in some
    triangulated hyperbolic manifolds when picking equal weights for
    the fixed points computed by r13_fixed_points_of_psl2c_matrix).
    """
    line = line_with_matrix.r13_line
    RF = line.points[0][0].parent()
    bias = RF(constants.start_point_bias)
    return line.points[0] + bias * line.points[1]