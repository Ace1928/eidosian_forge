import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def is_root(G, u, edgekeys):
    """
            Returns True if `u` is a root node in G.

            Node `u` will be a root node if its in-degree, restricted to the
            specified edges, is equal to 0.

            """
    if u not in G:
        raise Exception(f'{u!r} not in G')
    for v in G.pred[u]:
        for edgekey in G.pred[u][v]:
            if edgekey in edgekeys:
                return (False, edgekey)
    else:
        return (True, None)