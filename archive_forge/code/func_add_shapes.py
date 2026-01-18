from .mcomplex_base import *
from .t3mlite import simplex
def add_shapes(self, shapes):
    """
        Given a shape for each tetrahedron, add to the tetrahedron.

        The only assumption made here about the type of shapes is that
        they support the basic arithmetic operations. In particular,
        they can be SnapPy numbers or complex intervals.
        """
    for tet, z in zip(self.mcomplex.Tetrahedra, shapes):
        zp = 1 / (1 - z)
        zpp = (z - 1) / z
        tet.ShapeParameters = {simplex.E01: z, simplex.E23: z, simplex.E02: zp, simplex.E13: zp, simplex.E03: zpp, simplex.E12: zpp}