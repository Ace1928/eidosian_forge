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
def are_same_tile(self, center1, center2):
    e = self.cuspTranslateEngine
    translated_center1 = e.translate_to_match(center1, center2)
    if not translated_center1:
        return False
    dist = translated_center1.dist(center2)
    if dist < self.baseTetInRadius:
        return True
    if dist > self.baseTetInRadius:
        return False
    raise InsufficientPrecisionError('When tiling, it could not be decided whether two given tiles are the same since the distance between their respective center cannot be verified to be either small or large enough. This can be avoided by increasing the precision.')