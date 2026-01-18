from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import Mcomplex, VERBOSE, edge_and_arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
def perform_moves(self, moves, tet_stop_num=0):
    """
        Assumes that three_to_two, two_to_three, etc. only rebuild the
        edge classes after each move and that they accept arrows as
        spec for the moves.
        """
    t = 0
    for move, edge, face, tet_index in moves:
        if len(self) <= tet_stop_num:
            break
        t = t + 1
        arrow = Arrow(edge, face, self[tet_index])
        move_fn = getattr(self, move)
        move_fn(arrow, must_succeed=True)
        self._relabel_tetrahedra()
    self.rebuild()