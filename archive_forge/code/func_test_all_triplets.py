import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_all_triplets():
    """Tests the all_triplets function."""
    G = nx.DiGraph()
    G.add_edges_from(['01', '02', '03', '04', '05', '12', '16', '51', '56', '65'])
    expected = [f'{i},{j},{k}' for i in range(7) for j in range(i + 1, 7) for k in range(j + 1, 7)]
    expected = [set(x.split(',')) for x in expected]
    actual = [set(x) for x in nx.all_triplets(G)]
    assert all((any((s1 == s2 for s1 in expected)) for s2 in actual))