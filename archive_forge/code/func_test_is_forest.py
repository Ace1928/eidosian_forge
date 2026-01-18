import pytest
import networkx as nx
def test_is_forest(self):
    assert nx.is_forest(self.T2)
    assert nx.is_forest(self.T3)
    assert nx.is_forest(self.T5)
    assert nx.is_forest(self.F1)
    assert nx.is_forest(self.N5)