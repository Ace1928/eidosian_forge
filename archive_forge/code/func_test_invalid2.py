import pytest
import networkx
def test_invalid2(self):
    pytest.raises((TypeError, networkx.NetworkXError), networkx.random_clustered_graph, [[1, 1], [1, 2], [0, 1]])