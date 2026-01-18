import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_is_isomorphic(self):
    assert iso.is_isomorphic(self.G1, self.G2)
    assert not iso.is_isomorphic(self.G1, self.G4)