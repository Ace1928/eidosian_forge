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
def find_tile(self, m):
    center = self.baseTetInCenter.translate_PGL(m)
    for key in self.keys(center):
        for tile in self.intervalTree.find(key):
            if self.are_same_tile(center, tile.center):
                return tile
    return None