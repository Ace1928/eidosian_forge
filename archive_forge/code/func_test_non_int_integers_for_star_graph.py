import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_non_int_integers_for_star_graph(self):
    np = pytest.importorskip('numpy')
    G = nx.star_graph(np.int32(3))
    assert len(G) == 4
    assert G.size() == 3