import pytest
import networkx as nx
def test_graph_disallowed(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.stochastic_graph(nx.Graph())