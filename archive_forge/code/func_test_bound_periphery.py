from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_bound_periphery(self):
    result = {1, 4, 13, 16}
    assert set(nx.periphery(self.G, usebounds=True)) == result