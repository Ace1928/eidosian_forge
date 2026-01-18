from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
def dist_path(v):
    paths = {v: [v]}
    _dijkstra(G, v, new_weight, paths=paths)
    return paths