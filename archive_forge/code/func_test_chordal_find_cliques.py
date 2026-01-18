import pytest
import networkx as nx
def test_chordal_find_cliques(self):
    cliques = {frozenset([9]), frozenset([7, 8]), frozenset([1, 2, 3]), frozenset([2, 3, 4]), frozenset([3, 4, 5, 6])}
    assert set(nx.chordal_graph_cliques(self.chordal_G)) == cliques
    with pytest.raises(nx.NetworkXError, match='Input graph is not chordal'):
        set(nx.chordal_graph_cliques(self.non_chordal_G))
    with pytest.raises(nx.NetworkXError, match='Input graph is not chordal'):
        set(nx.chordal_graph_cliques(self.self_loop_G))