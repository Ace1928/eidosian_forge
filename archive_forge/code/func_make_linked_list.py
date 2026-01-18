from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
@staticmethod
def make_linked_list(pieces):
    """
        Given a list of pieces, populates next_ and prev of each
        piece to turn it into a linked list.
        """
    n = len(pieces)
    for i in range(n):
        a = pieces[i]
        b = pieces[(i + 1) % n]
        a.next_ = b
        b.prev = a