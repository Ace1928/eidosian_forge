import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_multigraph2(self):
    G = nx.MultiGraph([(1, 2), (2, 1)])
    L = nx.line_graph(G)
    assert edges_equal(L.edges(), [((1, 2, 0), (1, 2, 1))])