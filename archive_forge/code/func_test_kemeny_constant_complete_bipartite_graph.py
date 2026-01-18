from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_complete_bipartite_graph(self):
    n1 = 5
    n2 = 4
    G = nx.complete_bipartite_graph(n1, n2)
    K = nx.kemeny_constant(G)
    assert np.isclose(K, n1 + n2 - 3 / 2)