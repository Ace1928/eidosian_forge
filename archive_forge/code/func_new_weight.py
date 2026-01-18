from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
def new_weight(u, v, d):
    return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]