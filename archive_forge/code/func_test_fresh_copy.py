import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_fresh_copy(self):
    G = self.Graph()
    G.add_node(0)
    G.add_edge(1, 2)
    self.add_attributes(G)
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges())
    assert len(G.nodes[0]) == 1
    ddict = G.adj[1][2][0] if G.is_multigraph() else G.adj[1][2]
    assert len(ddict) == 1
    assert len(H.nodes[0]) == 0
    ddict = H.adj[1][2][0] if H.is_multigraph() else H.adj[1][2]
    assert len(ddict) == 0