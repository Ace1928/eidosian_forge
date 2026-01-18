import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nested_s_blossom_relabel_expand(self):
    """Create nested S-blossom, relabel as T, expand:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 19), (1, 3, 20), (1, 8, 8), (2, 3, 25), (2, 4, 18), (3, 5, 18), (4, 5, 13), (4, 7, 7), (5, 6, 7)])
    answer = matching_dict_to_set({1: 8, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4, 8: 1})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)