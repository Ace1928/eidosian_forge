from . import constants
from . import exceptions
from . import epsilons
from .line import distance_r13_lines, R13Line, R13LineWithMatrix
from .geodesic_info import GeodesicInfo, LiftedTetrahedron
from .quotient_space import balance_end_points_of_line, ZQuotientLiftedTetrahedronSet
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
import heapq
from typing import Sequence, Any
def add_structures_necessary_for_tube(mcomplex: Mcomplex) -> None:
    """
    A GeodesicTube can only be built from an Mcomplex if add_r13_geometry
    and this function (add_structure_necessary_for_tube) was called.

    This function adds R13Line objects for the edges of the tetrahedra.
    It also adds a bounding plane for each edge of each face of each
    tetrahedron. Such a bounding plane is perpendicular to the plane supporting
    the face and intersects the plane in an edge of face. That is, the
    bounding planes for a face cut out the triangle in the plane supporting
    the face.

    This information is used to compute the distance (or at least a lower bound
    for the distance) of a hyperbolic line L to a (triangular) face of a
    tetrahedron.

    In particular, we can check whether both endpoints of L fall "outside" of
    one of the bounding planes. In that case, the point of the triangle
    closest to the line is on edge corresponding to the bounding plane.
    """
    for tet in mcomplex.Tetrahedra:
        tet.R13_edges = {e: R13Line([tet.R13_vertices[simplex.Head[e]], tet.R13_vertices[simplex.Tail[e]]]) for e in simplex.OneSubsimplices}
        tet.triangle_bounding_planes = {f: {e: triangle_bounding_plane(tet, f, e) for e in _face_to_edges[f]} for f in simplex.TwoSubsimplices}