from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_ancestors_descendants_undirected():
    """Regression test to ensure ancestors and descendants work as expected on
    undirected graphs."""
    G = nx.path_graph(5)
    nx.ancestors(G, 2) == nx.descendants(G, 2) == {0, 1, 3, 4}