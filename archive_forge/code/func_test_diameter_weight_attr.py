from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_diameter_weight_attr(self):
    assert nx.diameter(self.G, weight='weight') == nx.diameter(self.G, weight='cost') == 1.6 != nx.diameter(self.G, weight='high_cost')