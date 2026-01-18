from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_center_weight_attr(self):
    center = set(nx.center(self.G, weight='weight'))
    assert center == set(nx.center(self.G, weight='cost')) != set(nx.center(self.G, weight='high_cost'))
    for v in center:
        assert nx.eccentricity(self.G, v, weight='high_cost') != pytest.approx(nx.eccentricity(self.G, v, weight='weight')) == pytest.approx(nx.eccentricity(self.G, v, weight='cost')) == nx.radius(self.G, weight='weight') == nx.radius(self.G, weight='cost') != nx.radius(self.G, weight='high_cost')
        assert nx.eccentricity(self.G, v, weight='high_cost') == nx.radius(self.G, weight='high_cost')