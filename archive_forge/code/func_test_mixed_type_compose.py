import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_mixed_type_compose():
    with pytest.raises(nx.NetworkXError):
        G = nx.Graph()
        H = nx.MultiGraph()
        I = nx.Graph()
        U = nx.compose_all([G, H, I])
    with pytest.raises(nx.NetworkXError):
        X = nx.Graph()
        Y = nx.DiGraph()
        XY = nx.compose_all([X, Y])