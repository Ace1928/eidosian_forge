import math
import pytest
import networkx as nx
def test_bad_beta_number(self):
    with pytest.raises(nx.NetworkXException):
        G = nx.Graph([(0, 1)])
        nx.katz_centrality(G, 0.1, beta='foo')