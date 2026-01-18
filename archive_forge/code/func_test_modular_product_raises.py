import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_modular_product_raises():
    G = nx.Graph([(0, 1), (1, 2), (2, 0)])
    H = nx.Graph([(0, 1), (1, 2), (2, 0)])
    DG = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    DH = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(G, DH)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(DG, H)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(DG, DH)
    MG = nx.MultiGraph([(0, 1), (1, 2), (2, 0), (0, 1)])
    MH = nx.MultiGraph([(0, 1), (1, 2), (2, 0), (0, 1)])
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(G, MH)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(MG, H)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(MG, MH)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.modular_product(nx.MultiGraph(G), H)