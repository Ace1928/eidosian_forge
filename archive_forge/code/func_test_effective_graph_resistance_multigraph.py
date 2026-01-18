from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_effective_graph_resistance_multigraph(self):
    G = nx.MultiGraph()
    G.add_edge(1, 2, weight=2)
    G.add_edge(1, 3, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(2, 3, weight=3)
    RG = nx.effective_graph_resistance(G, 'weight', True)
    edge23 = 1 / (1 / 1 + 1 / 3)
    rd12 = 1 / (1 / (1 + edge23) + 1 / 2)
    rd13 = 1 / (1 / (1 + 2) + 1 / edge23)
    rd23 = 1 / (1 / (2 + edge23) + 1 / 1)
    assert np.isclose(RG, rd12 + rd13 + rd23)