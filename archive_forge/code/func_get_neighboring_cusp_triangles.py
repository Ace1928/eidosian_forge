from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq
def get_neighboring_cusp_triangles(self, cusp_triangle):
    tet_index, V, m = cusp_triangle
    tet = self.original_mcomplex.Tetrahedra[tet_index]
    for F in simplex.TwoSubsimplices:
        if simplex.is_subset(V, F):
            yield (tet.Neighbor[F].Index, tet.Gluing[F].image(V), m * self.get_matrix_for_tet_and_face(tet, F))