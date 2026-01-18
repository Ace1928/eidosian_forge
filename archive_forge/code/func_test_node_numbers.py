from pytest import approx
from networkx import is_connected, neighbors
from networkx.generators.internet_as_graphs import random_internet_as_graph
def test_node_numbers(self):
    assert len(self.G.nodes()) == self.n
    assert len(self.T) < 7
    assert len(self.M) == round(self.n * 0.15)
    assert len(self.CP) == round(self.n * 0.05)
    numb = self.n - len(self.T) - len(self.M) - len(self.CP)
    assert len(self.C) == numb