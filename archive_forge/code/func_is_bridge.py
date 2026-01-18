from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def is_bridge(self):
    """
        Returns whether the link is in bridge position with respect to this
        height function.
        """
    return all((i == j for i, j in enumerate(sorted(self.snake_pos.values()))))