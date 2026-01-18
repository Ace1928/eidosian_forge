import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_directed_core_number(self):
    """core number had a bug for directed graphs found in issue #1959"""
    G = nx.DiGraph()
    edges = [(1, 2), (2, 1), (2, 3), (2, 4), (3, 4), (4, 3)]
    G.add_edges_from(edges)
    assert nx.core_number(G) == {1: 2, 2: 2, 3: 2, 4: 2}
    more_edges = [(1, 5), (3, 5), (4, 5), (3, 6), (4, 6), (5, 6)]
    G.add_edges_from(more_edges)
    assert nx.core_number(G) == {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}