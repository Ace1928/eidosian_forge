import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_generic_multitype():
    nxg = nx.graphviews
    G = nx.DiGraph([(1, 2)])
    with pytest.raises(nx.NetworkXError):
        nxg.generic_graph_view(G, create_using=nx.MultiGraph)
    G = nx.MultiDiGraph([(1, 2)])
    with pytest.raises(nx.NetworkXError):
        nxg.generic_graph_view(G, create_using=nx.DiGraph)