import pytest
import networkx as nx
@pytest.mark.parametrize('G', (nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()))
def test_is_chordal_not_implemented(self, G):
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.is_chordal(G)