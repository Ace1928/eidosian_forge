import pytest
import networkx as nx
def test_connected_watts_strogatz_zero_tries():
    with pytest.raises(nx.NetworkXError, match='Maximum number of tries exceeded'):
        nx.connected_watts_strogatz_graph(10, 2, 0.1, tries=0)