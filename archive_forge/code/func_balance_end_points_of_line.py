from .spatial_dict import SpatialDict, floor_as_integers
from .line import R13Line, R13LineWithMatrix
from .geodesic_info import LiftedTetrahedron
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
def balance_end_points_of_line(line_with_matrix: R13LineWithMatrix, point) -> R13LineWithMatrix:
    return R13LineWithMatrix(R13Line([endpoint / -r13_dot(point, endpoint) for endpoint in line_with_matrix.r13_line.points]), line_with_matrix.o13_matrix)