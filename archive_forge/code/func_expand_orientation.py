from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def expand_orientation(cs, kind):
    c, i = cs
    kinds = CyclicList(['up', 'down', 'down', 'up'])
    if c.kind in 'horizontal':
        s = 0 if i in [0, 3] else 2
    elif c.kind == 'vertical':
        s = 1 if i in [2, 3] else 3
    if kind in ['down', 'max']:
        s += 2
    return [(CrossingStrand(c, i), kinds[i + s]) for i in range(4)]