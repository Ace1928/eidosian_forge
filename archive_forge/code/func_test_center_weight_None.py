from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_center_weight_None(self):
    for v in set(nx.center(self.G, weight=None)):
        assert pytest.approx(nx.eccentricity(self.G, v, weight=None)) == nx.radius(self.G, weight=None)