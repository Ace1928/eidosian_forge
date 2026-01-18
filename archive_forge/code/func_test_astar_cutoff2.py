import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_cutoff2(self):
    assert nx.astar_path(self.XG, 's', 'v', cutoff=10.0) == ['s', 'x', 'u', 'v']
    assert nx.astar_path_length(self.XG, 's', 'v') == 9