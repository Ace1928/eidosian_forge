from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_center(self):
    assert set(nx.center(self.G)) == {6, 7, 10, 11}