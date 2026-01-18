from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_compute_v_structures():
    edges = [(0, 1), (0, 2), (3, 2)]
    G = nx.DiGraph(edges)
    v_structs = set(nx.compute_v_structures(G))
    assert len(v_structs) == 1
    assert (0, 2, 3) in v_structs
    edges = [('A', 'B'), ('C', 'B'), ('B', 'D'), ('D', 'E'), ('G', 'E')]
    G = nx.DiGraph(edges)
    v_structs = set(nx.compute_v_structures(G))
    assert len(v_structs) == 2