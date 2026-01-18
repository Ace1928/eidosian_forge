import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def edmonds_find_desired_edge(v):
    """
        Find the edge directed towards v with maximal weight.

        If an edge partition exists in this graph, return the included
        edge if it exists and never return any excluded edge.

        Note: There can only be one included edge for each vertex otherwise
        the edge partition is empty.

        Parameters
        ----------
        v : node
            The node to search for the maximal weight incoming edge.
        """
    edge = None
    max_weight = -INF
    for u, _, key, data in G.in_edges(v, data=True, keys=True):
        if data.get(partition) == nx.EdgePartition.EXCLUDED:
            continue
        new_weight = data[attr]
        if data.get(partition) == nx.EdgePartition.INCLUDED:
            max_weight = new_weight
            edge = (u, v, key, new_weight, data)
            break
        if new_weight > max_weight:
            max_weight = new_weight
            edge = (u, v, key, new_weight, data)
    return (edge, max_weight)