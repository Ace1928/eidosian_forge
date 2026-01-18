import pytest
import networkx as nx
def test_directed_configuration_raise_unequal():
    with pytest.raises(nx.NetworkXError):
        zin = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1]
        zout = [5, 3, 3, 3, 3, 2, 2, 2, 1, 2]
        nx.directed_configuration_model(zin, zout)