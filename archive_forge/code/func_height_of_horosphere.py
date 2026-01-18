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
def height_of_horosphere(self, vertex, is_at_infinity):
    if is_at_infinity ^ (vertex in self.vertices_at_infinity):
        raise Exception('An inconsistency was encountered showing that there is a bug in the code. Please report the example leading to this exception.')
    pts, cusp_length = self._horo_intersection_data(vertex, is_at_infinity)
    if is_at_infinity:
        base_length = abs(pts[2] - pts[1])
        return base_length / cusp_length
    else:

        def invDiff(a, b):
            if a == Infinity:
                return 0
            return 1 / (a - b)
        base_length = abs(invDiff(pts[2], pts[0]) - invDiff(pts[1], pts[0]))
        return cusp_length / base_length