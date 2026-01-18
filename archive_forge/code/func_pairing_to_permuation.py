from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def pairing_to_permuation(pairing):
    points = sorted(sum(pairing, tuple()))
    assert points == list(range(len(points)))
    ans = len(points) * [None]
    for x, y in pairing:
        ans[x], ans[y] = (y, x)
    return ans