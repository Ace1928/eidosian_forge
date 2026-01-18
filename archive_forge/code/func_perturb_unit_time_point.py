from . import constants
from . import epsilons
from . import exceptions
from .geodesic_tube import add_structures_necessary_for_tube, GeodesicTube
from .geodesic_info import GeodesicInfo
from .line import R13Line, distance_r13_lines
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import vector # type: ignore
from ..math_basics import correct_min # type: ignore
from typing import Sequence, List
def perturb_unit_time_point(point, max_amt, verified: bool):
    RF = point.base_ring()
    amt = RF(0.5) * max_amt
    direction = vector([RF(x) for x in constants.point_perturbation_direction])
    perturbed_origin = vector([amt.cosh()] + list(amt.sinh() * direction.normalized()))
    m = unit_time_vector_to_o13_hyperbolic_translation(point)
    perturbed_point = m * perturbed_origin
    if not verified:
        return perturbed_point
    space_coords = [RF(x.center()) for x in perturbed_point[1:4]]
    time_coord = sum((x ** 2 for x in space_coords), RF(1)).sqrt()
    perturbed_point = vector([time_coord] + space_coords)
    d = distance_unit_time_r13_points(point, perturbed_point)
    if not d < RF(0.75) * max_amt:
        raise InsufficientPrecisionError('Could not verify perturbed point is close enough to original start point. Increasing the precision will probably fix this.')
    return perturbed_point