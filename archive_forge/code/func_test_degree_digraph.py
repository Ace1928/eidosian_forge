import pytest
import networkx
import networkx as nx
from .historical_tests import HistoricalTests
def test_degree_digraph(self):
    H = nx.DiGraph()
    H.add_edges_from([(1, 24), (1, 2)])
    assert sorted((d for n, d in H.in_degree([1, 24]))) == [0, 1]
    assert sorted((d for n, d in H.out_degree([1, 24]))) == [0, 2]
    assert sorted((d for n, d in H.degree([1, 24]))) == [1, 2]