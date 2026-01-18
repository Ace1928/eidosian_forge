import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_non_extreme_probability_value(self):
    G = duplication_divergence_graph(6, p=0.3, seed=42)
    assert len(G) == 6
    assert list(G.degree()) == [(0, 2), (1, 3), (2, 2), (3, 3), (4, 1), (5, 1)]