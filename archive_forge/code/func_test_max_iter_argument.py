from itertools import permutations
import pytest
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
def test_max_iter_argument():
    G = nx.Graph([('A', 'B', {'weight': 1}), ('A', 'C', {'weight': 2}), ('A', 'D', {'weight': 3}), ('A', 'E', {'weight': 2}), ('A', 'F', {'weight': 4}), ('B', 'C', {'weight': 1}), ('B', 'D', {'weight': 4}), ('B', 'E', {'weight': 2}), ('B', 'F', {'weight': 1}), ('C', 'D', {'weight': 3}), ('C', 'E', {'weight': 2}), ('C', 'F', {'weight': 1}), ('D', 'E', {'weight': 4}), ('D', 'F', {'weight': 3}), ('E', 'F', {'weight': 2})])
    partition = ({'A', 'B', 'C'}, {'D', 'E', 'F'})
    C = kernighan_lin_bisection(G, partition, max_iter=1)
    assert_partition_equal(C, ({'A', 'F', 'C'}, {'D', 'E', 'B'}))