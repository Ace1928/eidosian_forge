import pytest
import networkx
import networkx as nx
from .historical_tests import HistoricalTests
def test_in_degree(self):
    G = self.G()
    G.add_nodes_from('GJK')
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'C'), ('C', 'D')])
    assert sorted((d for n, d in G.in_degree())) == [0, 0, 0, 0, 1, 2, 2]
    assert dict(G.in_degree()) == {'A': 0, 'C': 2, 'B': 1, 'D': 2, 'G': 0, 'K': 0, 'J': 0}