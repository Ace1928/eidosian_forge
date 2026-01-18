from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
@pytest.mark.parametrize('dod', raise_cases)
def test_non_multigraph_input_raise(self, dod):
    pytest.raises(nx.NetworkXError, self.Graph, dod, multigraph_input=True)
    pytest.raises(nx.NetworkXError, nx.to_networkx_graph, dod, create_using=self.Graph, multigraph_input=True)