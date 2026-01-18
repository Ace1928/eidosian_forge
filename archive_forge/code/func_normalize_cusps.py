from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def normalize_cusps(self, areas=None):
    """
        Scale cusp so that they have the given target area.
        Without argument, each cusp is scaled to have area 1.
        If the argument is a number, scale each cusp to have that area.
        If the argument is an array, scale each cusp by the respective
        entry in the array.
        """
    current_areas = self.cusp_areas()
    if not areas:
        areas = [1 for area in current_areas]
    elif not isinstance(areas, list):
        areas = [areas for area in current_areas]
    scales = [sqrt(area / current_area) for area, current_area in zip(areas, current_areas)]
    self.scale_cusps(scales)