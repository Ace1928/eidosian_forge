import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def subgraph_is_isomorphic(self, symmetry=False):
    """
        Returns True if a subgraph of :attr:`graph` is isomorphic to
        :attr:`subgraph` and False otherwise.

        Returns
        -------
        bool
        """
    isom = next(self.subgraph_isomorphisms_iter(symmetry=symmetry), None)
    return isom is not None