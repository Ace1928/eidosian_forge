from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def casson_split(self, manifold):
    """

        Returns the "Casson Split" of the manifold along the normal
        surface.  That is, splits the manifold open along the surface
        and replaces the "combinatorial I-bundles" by I-bundles over
        disks.  Of course, doing so may change the topology of
        complementary manifold.

        """
    M = manifold
    have_quads = [self.has_quad(i) for i in range(len(M))]
    new_tets = {}
    for i in have_quads:
        new_tets[i] = Tetrahedron()
    for i in have_quads:
        T = new_tets[i]