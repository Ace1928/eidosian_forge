import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nested_s_blossom_expand(self):
    """Create nested S-blossom, augment, expand recursively:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 8), (1, 3, 8), (2, 3, 10), (2, 4, 12), (3, 5, 12), (4, 5, 14), (4, 6, 12), (5, 7, 12), (6, 7, 14), (7, 8, 12)])
    answer = matching_dict_to_set({1: 2, 2: 1, 3: 5, 4: 6, 5: 3, 6: 4, 7: 8, 8: 7})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)