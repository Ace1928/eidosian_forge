import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_all_triads():
    """Tests the all_triplets function."""
    G = nx.DiGraph()
    G.add_edges_from(['01', '02', '03', '04', '05', '12', '16', '51', '56', '65'])
    expected = [f'{i},{j},{k}' for i in range(7) for j in range(i + 1, 7) for k in range(j + 1, 7)]
    expected = [G.subgraph(x.split(',')) for x in expected]
    actual = list(nx.all_triads(G))
    assert all((any((nx.is_isomorphic(G1, G2) for G1 in expected)) for G2 in actual))