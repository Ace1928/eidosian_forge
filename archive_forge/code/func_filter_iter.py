from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise
def filter_iter(nodes):

    def iterate(v):
        for w in nodes(v):
            if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                yield w
    return iterate