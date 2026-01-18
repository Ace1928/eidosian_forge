import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_eppstein_matching(self):
    """Tests that David Eppstein's implementation of the Hopcroft--Karp
        algorithm produces a maximum cardinality matching.

        """
    self.check_match(eppstein_matching(self.graph, self.top_nodes))