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
def replace_by(start_piece, end_piece, pieces) -> None:
    """
        Replaces the pieces between start_piece and end_piece (inclusive)
        by the given (not linked) list of pieces in the linked list that
        start_piece and end_piece participate in.
        """
    if start_piece.prev is end_piece:
        items = pieces + [pieces[0]]
    else:
        items = [start_piece.prev] + pieces + [end_piece.next_]
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]
        a.next_ = b
        b.prev = a
    for piece in [start_piece, end_piece]:
        if piece.tracker:
            piece.tracker.set_geodesic_piece(pieces[0])
            break