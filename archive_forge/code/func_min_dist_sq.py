import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def min_dist_sq(points):
    pairs = itertools.combinations(points, 2)
    return min((norm_sq(p - q) for p, q in pairs))