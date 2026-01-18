from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def init_vertices(self):
    """
        Computes vertices for the initial tetrahedron such that vertex 0, 1
        and 2 are at Infinity, 0 and z.
        """
    tet = self.mcomplex.ChooseGenInitialTet
    z = tet.ShapeParameters[simplex.E01]
    CF = z.parent()
    return {simplex.V0: Infinity, simplex.V1: CF(0), simplex.V2: CF(1), simplex.V3: z}