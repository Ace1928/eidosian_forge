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
def get_matrix_for_tet_and_face(self, tet, F):
    g = tet.GeneratorsInfo[F]
    if g == 0:
        tet = self.mcomplex.Tetrahedra[0]
        CIF = tet.ShapeParameters[simplex.E01].parent()
        return matrix.identity(CIF, 2)
    return self.mcomplex.GeneratorMatrices[g]