import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_intersection_all_multigraph_attributes():
    g = nx.MultiGraph()
    g.add_edge(0, 1, key=0)
    g.add_edge(0, 1, key=1)
    g.add_edge(0, 1, key=2)
    h = nx.MultiGraph()
    h.add_edge(0, 1, key=0)
    h.add_edge(0, 1, key=3)
    gh = nx.intersection_all([g, h])
    assert set(gh.nodes()) == set(g.nodes())
    assert set(gh.nodes()) == set(h.nodes())
    assert sorted(gh.edges()) == [(0, 1)]
    assert sorted(gh.edges(keys=True)) == [(0, 1, 0)]