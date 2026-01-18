import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_hopcroft_karp_matching_simple(self):
    match = hopcroft_karp_matching(self.simple_graph)
    assert match == self.simple_solution