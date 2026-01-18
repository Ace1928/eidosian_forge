import pytest
import networkx as nx
def test_is_chordal(self):
    assert not nx.is_chordal(self.non_chordal_G)
    assert nx.is_chordal(self.chordal_G)
    assert nx.is_chordal(self.connected_chordal_G)
    assert nx.is_chordal(nx.Graph())
    assert nx.is_chordal(nx.complete_graph(3))
    assert nx.is_chordal(nx.cycle_graph(3))
    assert not nx.is_chordal(nx.cycle_graph(5))
    assert nx.is_chordal(self.self_loop_G)