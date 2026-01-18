import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_biconnected_components_cycle():
    G = nx.cycle_graph(3)
    nx.add_cycle(G, [1, 3, 4])
    answer = [{0, 1, 2}, {1, 3, 4}]
    assert_components_equal(list(nx.biconnected_components(G)), answer)