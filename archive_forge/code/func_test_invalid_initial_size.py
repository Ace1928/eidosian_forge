import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_invalid_initial_size(self):
    with pytest.raises(NetworkXError):
        N = 5
        n = 10
        p = 0.5
        q = 0.5
        G = partial_duplication_graph(N, n, p, q)