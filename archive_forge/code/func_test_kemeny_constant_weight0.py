from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_kemeny_constant_weight0(self):
    G = nx.Graph()
    w12 = 0
    w13 = 3
    w23 = 4
    G.add_edge(1, 2, weight=w12)
    G.add_edge(1, 3, weight=w13)
    G.add_edge(2, 3, weight=w23)
    K = nx.kemeny_constant(G, weight='weight')
    test_data = 3 / 2 * (w12 + w13) * (w12 + w23) * (w13 + w23) / (w12 ** 2 * (w13 + w23) + w13 ** 2 * (w12 + w23) + w23 ** 2 * (w12 + w13) + 3 * w12 * w13 * w23)
    assert np.isclose(K, test_data)