import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_intersection_all_digraph():
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 3)])
    h = nx.DiGraph()
    h.add_edges_from([(2, 1), (2, 3)])
    gh = nx.intersection_all([g, h])
    assert sorted(gh.edges()) == [(2, 3)]