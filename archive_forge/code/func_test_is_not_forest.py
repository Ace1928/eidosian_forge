import pytest
import networkx as nx
def test_is_not_forest(self):
    assert not nx.is_forest(self.N4)
    assert not nx.is_forest(self.N6)
    assert not nx.is_forest(self.NF1)