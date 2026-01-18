import pytest
import networkx as nx
def test_directed_graph_raises_error(self):
    with pytest.raises(nx.NetworkXError, match='Directed Graph not supported'):
        nx.random_clustered_graph([(1, 2), (2, 1), (1, 1), (1, 1), (1, 1), (2, 0)], create_using=nx.DiGraph)