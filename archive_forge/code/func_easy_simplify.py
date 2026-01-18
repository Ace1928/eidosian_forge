from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def easy_simplify(self):
    """
        Perform moves eliminating edges of valence 1, 2, and 3,
        monotonically reducing the number of tetrahedra until no
        further such moves are possible.  Returns whether or not the
        number of tetrahedra was reduced.

        >>> M = Mcomplex('zLALvwvMwLzzAQPQQkbcbeijmoomvwuvust'
        ...              'wwytxtyxyahkswpmakguadppmrssxbkoxsi')
        >>> M.easy_simplify()
        True
        >>> len(M)
        1
        >>> M.rebuild(); M.isosig()
        'bkaagj'
        """
    init_tet = len(self)
    progress = True
    while progress:
        curr_tet = len(self)
        while self.attack_valence_one():
            pass
        while self.eliminate_valence_two() | self.eliminate_valence_three():
            pass
        progress = len(self) < curr_tet
    return len(self) < init_tet