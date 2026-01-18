from .line import R13LineWithMatrix
from . import epsilons
from . import constants
from . import exceptions
from ..hyperboloid import r13_dot, o13_inverse, distance_unit_time_r13_points # type: ignore
from ..snap.t3mlite import simplex # type: ignore
from ..snap.t3mlite import Tetrahedron, Vertex, Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import matrix # type: ignore
from typing import Tuple, Sequence, Optional, Any
def find_tet_or_core_curve(self) -> None:
    """
        Apply Deck-transformations to the start and end point and hyperbolic
        line until we either detected that the given geodesic corresponds to
        a core curve (only if line is not None) or we have captured the start
        point in one or two tetrahedra (in case the start close is on or very
        close to a face).
        This method also computes the distance of the geodesic to the core
        curves (only if line is not None) and raises an exception if we could
        not ensure that this distance is positive.
        """
    self.tet = None
    self.lifted_tetrahedra = ()
    self.core_curve_cusp = None
    self.core_curve_direction = 0
    tet, faces, cusp_curve_vertex = self._graph_trace(self.mcomplex.baseTet)
    if cusp_curve_vertex is not None:
        self.core_curve_direction = self._verify_direction_of_core_curve(tet, cusp_curve_vertex)
        self.core_curve_cusp = tet.Class[cusp_curve_vertex]
        return
    id_matrix = matrix.identity(ring=self.mcomplex.RF, n=4)
    if len(faces) == 0:
        self.tet = tet
        self.lifted_tetrahedra = [LiftedTetrahedron(tet, id_matrix)]
        return
    if len(faces) == 1:
        face, = faces
        other_tet = tet.Neighbor[face]
        other_unnormalised_start_point = tet.O13_matrices[face] * self.unnormalised_start_point
        other_face = tet.Gluing[face].image(face)
        for f in simplex.TwoSubsimplices:
            if f != other_face:
                if not r13_dot(other_unnormalised_start_point, other_tet.R13_planes[f]) < 0:
                    raise InsufficientPrecisionError('Failed to find lift of geodesic and prove that it intersects tetrahedra of the fundamental domain. Increasing the precision will probably fix this problem.')
        self.lifted_tetrahedra = [LiftedTetrahedron(tet, id_matrix), LiftedTetrahedron(other_tet, other_tet.O13_matrices[other_face])]
        return
    raise InsufficientPrecisionError('Start point chosen on geodesic too close to 1-skeleton of triangulation to verify it is not on the 1-skeleton. Increasing the precision will probably fix this problem.')