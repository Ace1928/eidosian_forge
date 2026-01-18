import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_different_order3(self):
    G1 = nx.complete_graph(7)
    G2 = nx.complete_graph(8)
    assert not vf2pp_is_isomorphic(G1, G2)