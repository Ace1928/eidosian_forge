import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def subgraph_isomorphisms_iter(self, symmetry=True):
    """Alternative name for :meth:`find_isomorphisms`."""
    return self.find_isomorphisms(symmetry)