from itertools import permutations
import pytest
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
def assert_partition_equal(x, y):
    assert set(map(frozenset, x)) == set(map(frozenset, y))