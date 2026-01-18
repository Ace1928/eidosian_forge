import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def sorted_edges(G, attr='weight', default=1):
    edges = [(u, v, data.get(attr, default)) for u, v, data in G.edges(data=True)]
    edges = sorted(edges, key=lambda x: (x[2], x[1], x[0]))
    return edges