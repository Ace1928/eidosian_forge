import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_is_attracting_component(self):
    assert not nx.is_attracting_component(self.G1)
    assert not nx.is_attracting_component(self.G2)
    assert not nx.is_attracting_component(self.G3)
    g2 = self.G3.subgraph([1, 2])
    assert nx.is_attracting_component(g2)
    assert not nx.is_attracting_component(self.G4)