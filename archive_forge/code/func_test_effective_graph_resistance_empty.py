from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_effective_graph_resistance_empty(self):
    G = nx.Graph()
    with pytest.raises(nx.NetworkXError):
        nx.effective_graph_resistance(G)