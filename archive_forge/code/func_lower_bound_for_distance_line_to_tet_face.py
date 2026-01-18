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
def lower_bound_for_distance_line_to_tet_face(line, tet, face, verified):
    RF = line.points[0][0].parent()
    if verified:
        epsilon = 0
    else:
        epsilon = epsilons.compute_epsilon(RF)
    a0 = r13_dot(tet.R13_planes[face], line.points[0])
    a1 = r13_dot(tet.R13_planes[face], line.points[1])
    abs0 = abs(a0)
    abs1 = abs(a1)
    if abs0 > epsilon and abs1 > epsilon:
        pt = line.points[0] / abs0 + line.points[1] / abs1
        for e in _face_to_edges[face]:
            if r13_dot(pt, tet.triangle_bounding_planes[face][e]) > epsilon:
                return distance_r13_lines(line, tet.R13_edges[e])
        p = a0 * a1
        if p > 0:
            return (-2 * p / line.inner_product).sqrt().arcsinh()
        return RF(0)
    else:
        for e in _face_to_edges[face]:
            p = tet.triangle_bounding_planes[face][e]
            b0 = r13_dot(line.points[0], p)
            b1 = r13_dot(line.points[1], p)
            if b0 > epsilon and b1 > epsilon:
                return distance_r13_lines(line, tet.R13_edges[e])
        return RF(0)