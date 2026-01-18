import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_eppstein_matching_disconnected(self):
    with pytest.raises(nx.AmbiguousSolution):
        match = eppstein_matching(self.disconnected_graph)