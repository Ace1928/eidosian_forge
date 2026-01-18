import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
@pytest.mark.parametrize('graph_type', (nx.Graph, nx.MultiGraph, nx.DiGraph))
def test_first_graph_empty(self, graph_type):
    G = graph_type()
    H = graph_type([(0, 1)])
    assert vf2pp_isomorphism(G, H) is None