import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_floating_point_weights(self):
    G = nx.Graph()
    G.add_edge(1, 2, weight=math.pi)
    G.add_edge(2, 3, weight=math.exp(1))
    G.add_edge(1, 3, weight=3.0)
    G.add_edge(1, 4, weight=math.sqrt(2.0))
    assert edges_equal(nx.max_weight_matching(G), matching_dict_to_set({1: 4, 2: 3, 3: 2, 4: 1}))
    assert edges_equal(nx.min_weight_matching(G), matching_dict_to_set({1: 4, 2: 3, 3: 2, 4: 1}))