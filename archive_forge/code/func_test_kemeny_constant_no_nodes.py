from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_no_nodes(self):
    G = nx.Graph()
    with pytest.raises(nx.NetworkXError):
        nx.kemeny_constant(G)