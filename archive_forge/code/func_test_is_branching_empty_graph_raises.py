import pytest
import networkx as nx
def test_is_branching_empty_graph_raises():
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXPointlessConcept, match='G has no nodes.'):
        nx.is_branching(G)