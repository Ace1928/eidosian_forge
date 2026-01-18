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
@staticmethod
def get_init_vertices(vertex):
    corner = vertex.Corners[0]
    v0, v1, v2, v3 = _OrientedVerticesForVertex[corner.Subsimplex]
    complex_lengths = corner.Tetrahedron.horotriangles[v0].lengths
    p2 = complex_lengths[v0 | v1 | v2]
    p3 = complex_lengths[v0 | v1 | v3]
    CIF = p2.parent()
    return {v0: Infinity, v1: CIF(0), v2: p2, v3: -p3}