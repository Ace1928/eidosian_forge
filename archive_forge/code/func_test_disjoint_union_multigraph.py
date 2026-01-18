import os
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.utils import edges_equal
def test_disjoint_union_multigraph():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0)
    G.add_edge(0, 1, key=1)
    H = nx.MultiGraph()
    H.add_edge(2, 3, key=0)
    H.add_edge(2, 3, key=1)
    GH = nx.disjoint_union(G, H)
    assert set(GH) == set(G) | set(H)
    assert set(GH.edges(keys=True)) == set(G.edges(keys=True)) | set(H.edges(keys=True))