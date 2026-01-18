import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nested_s_blossom(self):
    """Create nested S-blossom, use for augmentation:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 9), (1, 3, 9), (2, 3, 10), (2, 4, 8), (3, 5, 8), (4, 5, 10), (5, 6, 6)])
    dict_format = {1: 3, 2: 4, 3: 1, 4: 2, 5: 6, 6: 5}
    expected = {frozenset(e) for e in matching_dict_to_set(dict_format)}
    answer = {frozenset(e) for e in nx.max_weight_matching(G)}
    assert answer == expected
    answer = {frozenset(e) for e in nx.min_weight_matching(G)}
    assert answer == expected