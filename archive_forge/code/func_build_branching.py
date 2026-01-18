import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def build_branching(edges, double=False):
    G = nx.DiGraph()
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
        if double:
            G.add_edge(u + 9, v + 9, weight=weight)
    return G