from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_bound_periphery_weight_attr(self):
    result = {4, 5}
    assert set(nx.periphery(self.G, usebounds=True, weight='weight')) == set(nx.periphery(self.G, usebounds=True, weight='cost')) == result