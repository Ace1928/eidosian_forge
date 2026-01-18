from ...sage_helper import _within_sage
from ..upper_halfspace.finite_point import *
def round_to_nearest_integers(interval):
    r = interval.round()
    return list(range(int(r.lower()), int(r.upper()) + 1))