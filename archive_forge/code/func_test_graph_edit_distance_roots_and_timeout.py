import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance_roots_and_timeout(self):
    G0 = nx.star_graph(5)
    G1 = G0.copy()
    pytest.raises(ValueError, graph_edit_distance, G0, G1, roots=[2])
    pytest.raises(ValueError, graph_edit_distance, G0, G1, roots=[2, 3, 4])
    pytest.raises(nx.NodeNotFound, graph_edit_distance, G0, G1, roots=(9, 3))
    pytest.raises(nx.NodeNotFound, graph_edit_distance, G0, G1, roots=(3, 9))
    pytest.raises(nx.NodeNotFound, graph_edit_distance, G0, G1, roots=(9, 9))
    assert graph_edit_distance(G0, G1, roots=(1, 2)) == 0
    assert graph_edit_distance(G0, G1, roots=(0, 1)) == 8
    assert graph_edit_distance(G0, G1, roots=(1, 2), timeout=5) == 0
    assert graph_edit_distance(G0, G1, roots=(0, 1), timeout=5) == 8
    assert graph_edit_distance(G0, G1, roots=(0, 1), timeout=0.0001) is None
    pytest.raises(nx.NetworkXError, graph_edit_distance, G0, G1, timeout=0)