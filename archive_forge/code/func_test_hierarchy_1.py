import pytest
import networkx as nx
def test_hierarchy_1():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 1), (3, 4), (0, 4)])
    assert nx.flow_hierarchy(G) == 0.5