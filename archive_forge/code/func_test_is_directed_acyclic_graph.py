from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_directed_acyclic_graph(self):
    G = nx.generators.complete_graph(2)
    assert not nx.is_directed_acyclic_graph(G)
    assert not nx.is_directed_acyclic_graph(G.to_directed())
    assert not nx.is_directed_acyclic_graph(nx.Graph([(3, 4), (4, 5)]))
    assert nx.is_directed_acyclic_graph(nx.DiGraph([(3, 4), (4, 5)]))