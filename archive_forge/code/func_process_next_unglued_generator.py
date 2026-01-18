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
def process_next_unglued_generator(self):
    unglued_generator = heapq.heappop(self.unglued_generator_heapq)
    m = unglued_generator.tile.matrix * self.mcomplex.GeneratorMatrices[unglued_generator.g]
    if not self.find_tile(m):
        tile = self.create_tile(m)
        unglued_generators, unglued_vertices = self.unglued_generators_and_vertices_for_tile(tile)
        for g in unglued_generators:
            self.record_unglued_generator(tile, g)
        for vertex in unglued_vertices:
            self.account_horosphere_height(tile, vertex)