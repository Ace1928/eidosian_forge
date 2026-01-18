import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_mixed_nodetypes():
    G = nx.Graph()
    edgelist = [(0, 3, [('weight', 5)]), (0, '1', [('weight', 5)])]
    G.add_edges_from(edgelist)
    G = G.to_directed()
    x = branchings.minimum_spanning_arborescence(G)