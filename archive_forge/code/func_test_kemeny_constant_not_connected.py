from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_not_connected(self):
    self.G.add_node(5)
    with pytest.raises(nx.NetworkXError):
        nx.kemeny_constant(self.G)