from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def cusp_areas(self):
    """
        List of all cusp areas.
        """
    return [CuspCrossSectionBase._cusp_area(cusp) for cusp in self.mcomplex.Vertices]