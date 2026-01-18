import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_hopcroft_karp_matching(self):
    """Tests that the Hopcroft--Karp algorithm produces a maximum
        cardinality matching in a bipartite graph.

        """
    self.check_match(hopcroft_karp_matching(self.graph, self.top_nodes))