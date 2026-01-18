import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_is_regular4(self):
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    assert reg.is_regular(g)