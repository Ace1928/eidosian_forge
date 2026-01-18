from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
@pytest.mark.parametrize('dod, edges', mgi_none_cases)
def test_non_multigraph_input_mgi_none(self, dod, edges):
    G = self.Graph(dod)
    assert list(G.edges(keys=True, data=True)) == edges