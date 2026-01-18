from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
def greedily_find_independent_set(self, P):
    """Greedily find an independent set of nodes from a set of
        nodes P."""
    independent_set = []
    P = P[:]
    while P:
        v = P[0]
        independent_set.append(v)
        P = [w for w in P if v != w and (not self.G.has_edge(v, w))]
    return independent_set