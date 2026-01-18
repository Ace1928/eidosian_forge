import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_invalid_probabilities(self):
    N = 1
    n = 1
    for p, q in [(0.5, 2), (0.5, -1), (2, 0.5), (-1, 0.5)]:
        args = (N, n, p, q)
        pytest.raises(NetworkXError, partial_duplication_graph, *args)