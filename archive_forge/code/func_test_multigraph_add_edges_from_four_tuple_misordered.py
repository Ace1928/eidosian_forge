from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_multigraph_add_edges_from_four_tuple_misordered(self):
    """add_edges_from expects 4-tuples of the format (u, v, key, data_dict).

        Ensure 4-tuples of form (u, v, data_dict, key) raise exception.
        """
    G = nx.MultiGraph()
    with pytest.raises(TypeError):
        G.add_edges_from([(0, 1, {'color': 'red'}, 0)])