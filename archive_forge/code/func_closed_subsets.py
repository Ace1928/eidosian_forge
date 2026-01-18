import networkx as nx
from collections import deque
def closed_subsets(self):
    """
        Generator for all nonempty transitively closed subsets.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> len(list(P.closed_subsets()))
        7
        """
    for X in powerset(self.elements):
        if not X:
            continue
        pairwise_incomparable = True
        for x in X:
            if any((y in self.smaller[x] or y in self.larger[x] for y in X)):
                pairwise_incomparable = False
                break
        if pairwise_incomparable:
            yield self.closure(X)