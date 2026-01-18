import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_invalid_matching(self):
    G = nx.path_graph(4)
    assert not nx.is_matching(G, {(0, 1), (1, 2), (2, 3)})