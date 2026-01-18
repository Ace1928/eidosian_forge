from .spatial_dict import SpatialDict, floor_as_integers
from .line import R13Line, R13LineWithMatrix
from .geodesic_info import LiftedTetrahedron
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
def representatives(self, point):
    a = r13_dot(point, self._line.points[0])
    b = r13_dot(point, self._line.points[1])
    r = (a / b).log() / self._log_scale_factor
    return [self._power_cache.power(i) * point for i in floor_as_integers(r)]