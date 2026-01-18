from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def set_heights(self):
    """
        Assigns a height to each min/max and crossing of the diagram.
        """
    D = self.digraph()
    self.heights = basic_topological_numbering(D)