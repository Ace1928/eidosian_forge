import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_complete_2_partite_graph(self):
    """Tests that the complete 2-partite graph is the complete bipartite
        graph.

        """
    G = nx.complete_multipartite_graph(2, 3)
    H = nx.complete_bipartite_graph(2, 3)
    assert nodes_equal(G, H)
    assert edges_equal(G.edges(), H.edges())