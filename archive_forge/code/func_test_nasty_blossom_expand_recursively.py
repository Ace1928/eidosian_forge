import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_nasty_blossom_expand_recursively(self):
    """Create nested S-blossom, relabel as S, expand recursively:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 40), (1, 3, 40), (2, 3, 60), (2, 4, 55), (3, 5, 55), (4, 5, 50), (1, 8, 15), (5, 7, 30), (7, 6, 10), (8, 10, 10), (4, 9, 30)])
    ans = {1: 2, 2: 1, 3: 5, 4: 9, 5: 3, 6: 7, 7: 6, 8: 10, 9: 4, 10: 8}
    answer = matching_dict_to_set(ans)
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)