import pytest
import networkx as nx
from networkx.algorithms.centrality.subgraph_alg import (
def test_estrada_index(self):
    answer = 1041.2470334195475
    result = estrada_index(nx.karate_club_graph())
    assert answer == pytest.approx(result, abs=1e-07)