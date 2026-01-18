from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_directed(self):
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.kemeny_constant(G)