import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def edmonds_remove_node(G, edge_index, n):
    """
        Remove a node from the graph, updating the edge index to match.

        Parameters
        ----------
        G : MultiDiGraph
            The graph to remove an edge from.
        edge_index : dict
            A mapping from integers to the edges of the graph.
        n : node
            The node to remove from `G`.
        """
    keys = set()
    for keydict in G.pred[n].values():
        keys.update(keydict)
    for keydict in G.succ[n].values():
        keys.update(keydict)
    for key in keys:
        del edge_index[key]
    G.remove_node(n)