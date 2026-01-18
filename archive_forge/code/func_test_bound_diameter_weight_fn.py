from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_bound_diameter_weight_fn(self):
    assert nx.diameter(self.G, usebounds=True, weight=self.weight_fn) == 6