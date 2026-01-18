import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_hopcroft_karp_matching_disconnected(self):
    with pytest.raises(nx.AmbiguousSolution):
        match = hopcroft_karp_matching(self.disconnected_graph)