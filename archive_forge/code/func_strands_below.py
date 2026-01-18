from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def strands_below(self, crossing):
    """
        The two upward strands below the crossing.
        """
    kinds = self.orientations
    a = CrossingStrand(crossing, 0)
    b = a.rotate()
    upmin = set(['up', 'min'])
    test_a = kinds[a] in upmin
    while True:
        test_b = kinds[b] in upmin
        if test_a and test_b:
            return (a, b)
        a, b = (b, b.rotate())
        test_a = test_b