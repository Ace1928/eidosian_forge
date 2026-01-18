import math
import pytest
import networkx as nx
def test_multigraph_numpy(self):
    with pytest.raises(nx.NetworkXException):
        nx.eigenvector_centrality_numpy(nx.MultiGraph())