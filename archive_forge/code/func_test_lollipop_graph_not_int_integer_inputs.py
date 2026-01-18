import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_lollipop_graph_not_int_integer_inputs(self):
    np = pytest.importorskip('numpy')
    G = nx.lollipop_graph(np.int32(4), np.int64(3))
    assert len(G) == 7
    assert G.size() == 9