import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_isomorphism(self):
    g1 = nx.Graph()
    nx.add_cycle(g1, range(4))
    g2 = nx.Graph()
    nx.add_cycle(g2, range(4))
    g2.add_edges_from(list(zip(g2, range(4, 8))))
    ismags = iso.ISMAGS(g2, g1)
    assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == [{n: n for n in g1.nodes}]