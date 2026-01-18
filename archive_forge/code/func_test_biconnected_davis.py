import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_biconnected_davis():
    D = nx.davis_southern_women_graph()
    bcc = list(nx.biconnected_components(D))[0]
    assert set(D) == bcc
    assert len(list(nx.articulation_points(D))) == 0