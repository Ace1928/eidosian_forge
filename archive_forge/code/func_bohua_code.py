from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def bohua_code(self):
    b = self.width // 2
    ans = [b]
    ans += pairing_to_permuation(self.bottom)
    ans += [len(self.crossings)] + list(sum(self.crossings, tuple()))
    ans += pairing_to_permuation(self.top)
    return self.name + '\t' + ' '.join((repr(a) for a in ans))