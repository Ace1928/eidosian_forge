from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_center_weight_fn(self):
    for v in set(nx.center(self.G, weight=self.weight_fn)):
        assert nx.eccentricity(self.G, v, weight=self.weight_fn) == nx.radius(self.G, weight=self.weight_fn)