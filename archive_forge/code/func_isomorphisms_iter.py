import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def isomorphisms_iter(self, symmetry=True):
    """
        Does the same as :meth:`find_isomorphisms` if :attr:`graph` and
        :attr:`subgraph` have the same number of nodes.
        """
    if len(self.graph) == len(self.subgraph):
        yield from self.subgraph_isomorphisms_iter(symmetry=symmetry)