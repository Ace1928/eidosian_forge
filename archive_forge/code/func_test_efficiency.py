import networkx as nx
def test_efficiency(self):
    assert nx.efficiency(self.G2, 0, 1) == 1
    assert nx.efficiency(self.G2, 0, 2) == 1 / 2