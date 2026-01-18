import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize(('m', 'n'), [('ab', ''), ('abc', 'defg')])
def test_lollipop_graph_size_node_sequence(self, m, n):
    G = nx.lollipop_graph(m, n)
    assert nx.number_of_nodes(G) == len(m) + len(n)
    assert nx.number_of_edges(G) == len(m) * (len(m) - 1) / 2 + len(n)