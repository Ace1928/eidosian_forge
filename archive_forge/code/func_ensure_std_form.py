from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def ensure_std_form(self, allow_scaling_up=False):
    """
        Makes sure that the cusp neighborhoods intersect each tetrahedron
        in standard form by scaling the cusp neighborhoods down if necessary.
        """
    z = self.mcomplex.Tetrahedra[0].ShapeParameters[t3m.simplex.E01]
    RF = z.real().parent()
    if allow_scaling_up:
        area_scales = [[] for v in self.mcomplex.Vertices]
    else:
        area_scales = [[RF(1)] for v in self.mcomplex.Vertices]
    for tet in self.mcomplex.Tetrahedra:
        z = tet.ShapeParameters[t3m.simplex.E01]
        max_area = ComplexCuspCrossSection._lower_bound_max_area_triangle_for_std_form(z)
        for zeroSubsimplex, triangle in tet.horotriangles.items():
            area_scale = max_area / triangle.area
            vertex = tet.Class[zeroSubsimplex]
            area_scales[vertex.Index].append(area_scale)
    scales = [sqrt(correct_min(s)) for s in area_scales]
    self.scale_cusps(scales)