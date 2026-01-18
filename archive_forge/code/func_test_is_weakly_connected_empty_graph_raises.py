import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_is_weakly_connected_empty_graph_raises():
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXPointlessConcept, match='Connectivity is undefined'):
        nx.is_weakly_connected(G)