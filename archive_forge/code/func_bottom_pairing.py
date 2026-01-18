from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def bottom_pairing(snake):
    cs = snake[0]
    return tuple(sorted([to_index(cs), to_index(cs.opposite())]))