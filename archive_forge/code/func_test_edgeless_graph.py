import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_edgeless_graph(self):
    G = nx.empty_graph(5)
    with pytest.raises(nx.NetworkXError, match='edgeless graph'):
        nx.inverse_line_graph(G)