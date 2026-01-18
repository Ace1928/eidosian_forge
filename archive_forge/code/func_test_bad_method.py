import pytest
import networkx as nx
def test_bad_method(self):
    with pytest.raises(ValueError):
        G = nx.path_graph(2)
        nx.average_shortest_path_length(G, weight='weight', method='SPAM')