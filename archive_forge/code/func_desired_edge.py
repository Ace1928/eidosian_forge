import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def desired_edge(v):
    """
            Find the edge directed toward v with maximal weight.

            If an edge partition exists in this graph, return the included edge
            if it exists and no not return any excluded edges. There can only
            be one included edge for each vertex otherwise the edge partition is
            empty.
            """
    edge = None
    weight = -INF
    for u, _, key, data in G.in_edges(v, data=True, keys=True):
        if data.get(partition) == nx.EdgePartition.EXCLUDED:
            continue
        new_weight = data[attr]
        if data.get(partition) == nx.EdgePartition.INCLUDED:
            weight = new_weight
            edge = (u, v, key, new_weight, data)
            return (edge, weight)
        if new_weight > weight:
            weight = new_weight
            edge = (u, v, key, new_weight, data)
    return (edge, weight)