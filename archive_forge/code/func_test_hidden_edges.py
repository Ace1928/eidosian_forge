import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_hidden_edges(self):
    hide_edges = [(2, 3, 4), (2, 3, 3), (8, 7, 0), (222, 223, 0)]
    edges_gone = self.hide_edges_filter(hide_edges)
    G = self.gview(self.G, filter_edge=edges_gone)
    assert self.G.nodes == G.nodes
    if G.is_directed():
        assert self.G.edges - G.edges == {(2, 3, 4)}
        assert list(G[3]) == [4]
        assert list(G[2]) == [3]
        assert list(G.pred[3]) == [2]
        assert list(G.pred[2]) == [1]
        assert G.size() == 9
    else:
        assert self.G.edges - G.edges == {(2, 3, 4), (7, 8, 0)}
        assert list(G[3]) == [2, 4]
        assert list(G[2]) == [1, 3]
        assert G.size() == 8
    assert G.degree(3) == 3
    pytest.raises(KeyError, G.__getitem__, 221)
    pytest.raises(KeyError, G.__getitem__, 222)