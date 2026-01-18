from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def check_cusp_development_exactly(self):
    """
        Check that all side lengths of horo triangles are consistent.
        If the logarithmic edge equations are fulfilled, this implices
        that the all cusps are complete and thus the manifold is complete.
        """
    for tet0 in self.mcomplex.Tetrahedra:
        for vert0 in t3m.simplex.ZeroSubsimplices:
            for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
                tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
                side0 = tet0.horotriangles[vert0].lengths[face0]
                side1 = tet1.horotriangles[vert1].lengths[face1]
                if not side0 == side1 * self.HoroTriangle.direction_sign():
                    raise CuspDevelopmentExactVerifyError(side0, side1)