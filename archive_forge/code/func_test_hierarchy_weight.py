import pytest
import networkx as nx
def test_hierarchy_weight():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1, {'weight': 0.3}), (1, 2, {'weight': 0.1}), (2, 3, {'weight': 0.1}), (3, 1, {'weight': 0.1}), (3, 4, {'weight': 0.3}), (0, 4, {'weight': 0.3})])
    assert nx.flow_hierarchy(G, weight='weight') == 0.75