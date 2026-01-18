import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_articulation_points_repetitions():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    assert list(nx.articulation_points(G)) == [1]