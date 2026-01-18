import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_biconnected_eppstein():
    G1 = nx.Graph({0: [1, 2, 5], 1: [0, 5], 2: [0, 3, 4], 3: [2, 4, 5, 6], 4: [2, 3, 5, 6], 5: [0, 1, 3, 4], 6: [3, 4]})
    G2 = nx.Graph({0: [2, 5], 1: [3, 8], 2: [0, 3, 5], 3: [1, 2, 6, 8], 4: [7], 5: [0, 2], 6: [3, 8], 7: [4], 8: [1, 3, 6]})
    assert nx.is_biconnected(G1)
    assert not nx.is_biconnected(G2)
    answer_G2 = [{1, 3, 6, 8}, {0, 2, 5}, {2, 3}, {4, 7}]
    bcc = list(nx.biconnected_components(G2))
    assert_components_equal(bcc, answer_G2)