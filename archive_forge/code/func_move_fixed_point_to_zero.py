from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def move_fixed_point_to_zero(self):
    """
        Determines the fixed point of the holonomies for all
        incomplete cusps. Then moves the vertex positions of the
        corresponding cusp triangles so that the fixed point is at the
        origin.

        It also add the boolean v.is_complete to all vertices of the
        triangulation to mark whether the corresponding cusp is
        complete or not.
        """
    for cusp, cusp_info in zip(self.mcomplex.Vertices, self.manifold.cusp_info()):
        cusp.is_complete = cusp_info['complete?']
        if not cusp.is_complete:
            fixed_pt = self._compute_cusp_fixed_point(cusp)
            for corner in cusp.Corners:
                tet, vert = (corner.Tetrahedron, corner.Subsimplex)
                trig = tet.horotriangles[vert]
                trig.vertex_positions = {edge: position - fixed_pt for edge, position in trig.vertex_positions.items()}