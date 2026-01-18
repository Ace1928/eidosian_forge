import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_is_regular_empty_graph_raises():
    G = nx.Graph()
    with pytest.raises(nx.NetworkXPointlessConcept, match='Graph has no nodes'):
        nx.is_regular(G)