import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_maximal_but_not_perfect(self):
    G = nx.cycle_graph(4)
    G.add_edge(0, 4)
    G.add_edge(1, 4)
    assert not nx.is_perfect_matching(G, {(1, 4), (0, 3)})