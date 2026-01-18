from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_aperiodic_bipartite():
    G = nx.DiGraph(nx.davis_southern_women_graph())
    assert not nx.is_aperiodic(G)