from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_multiple_roots(self):
    """Tests that a directed acyclic graph with multiple degree zero
        nodes creates an arborescence with multiple (weakly) connected
        components.

        """
    G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (5, 2)])
    B = nx.dag_to_branching(G)
    expected = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4), (5, 6), (6, 7)])
    assert nx.is_branching(B)
    assert not nx.is_arborescence(B)
    assert nx.is_isomorphic(B, expected)