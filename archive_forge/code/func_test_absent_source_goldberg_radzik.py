import pytest
import networkx as nx
from networkx.utils import pairwise
def test_absent_source_goldberg_radzik(self):
    with pytest.raises(nx.NodeNotFound):
        G = nx.path_graph(2)
        nx.goldberg_radzik(G, 3, 0)