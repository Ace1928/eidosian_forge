import pytest
import networkx as nx
from networkx.algorithms.centrality.subgraph_alg import (
def test_communicability_betweenness_centrality_small(self):
    result = communicability_betweenness_centrality(nx.path_graph(2))
    assert result == {0: 0, 1: 0}
    result = communicability_betweenness_centrality(nx.path_graph(1))
    assert result == {0: 0}
    result = communicability_betweenness_centrality(nx.path_graph(0))
    assert result == {}
    answer = {0: 0.1411224421177313, 1: 1.0, 2: 0.1411224421177313}
    result = communicability_betweenness_centrality(nx.path_graph(3))
    for k, v in result.items():
        assert answer[k] == pytest.approx(v, abs=1e-07)
    result = communicability_betweenness_centrality(nx.complete_graph(3))
    for k, v in result.items():
        assert 0.49786143366223296 == pytest.approx(v, abs=1e-07)