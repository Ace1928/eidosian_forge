import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nested_s_blossom_relabel(self):
    """Create S-blossom, relabel as S, include in nested S-blossom:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 10), (1, 7, 10), (2, 3, 12), (3, 4, 20), (3, 5, 20), (4, 5, 25), (5, 6, 10), (6, 7, 10), (7, 8, 8)])
    answer = matching_dict_to_set({1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)