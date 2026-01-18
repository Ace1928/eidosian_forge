from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
@pytest.mark.parametrize('dod, mgi, edges', cases)
def test_non_multigraph_input(self, dod, mgi, edges):
    G = self.Graph(dod, multigraph_input=mgi)
    assert list(G.edges(keys=True, data=True)) == edges
    G = nx.to_networkx_graph(dod, create_using=self.Graph, multigraph_input=mgi)
    assert list(G.edges(keys=True, data=True)) == edges