import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_attracting_components(self):
    ac = list(nx.attracting_components(self.G1))
    assert {2} in ac
    assert {9} in ac
    assert {10} in ac
    ac = list(nx.attracting_components(self.G2))
    ac = [tuple(sorted(x)) for x in ac]
    assert ac == [(1, 2)]
    ac = list(nx.attracting_components(self.G3))
    ac = [tuple(sorted(x)) for x in ac]
    assert (1, 2) in ac
    assert (3, 4) in ac
    assert len(ac) == 2
    ac = list(nx.attracting_components(self.G4))
    assert ac == []