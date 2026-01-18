import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_copy_of_view(self):
    G = nx.MultiGraph(self.MGv)
    assert G.__class__.__name__ == 'MultiGraph'
    G = G.copy(as_view=True)
    assert G.__class__.__name__ == 'MultiGraph'