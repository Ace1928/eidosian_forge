import pytest
import networkx as nx
def test_chordal_find_cliquesCC(self):
    cliques = {frozenset([1, 2, 3]), frozenset([2, 3, 4]), frozenset([3, 4, 5, 6])}
    cgc = nx.chordal_graph_cliques
    assert set(cgc(self.connected_chordal_G)) == cliques