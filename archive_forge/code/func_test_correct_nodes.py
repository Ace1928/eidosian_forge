import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_correct_nodes(self):
    """Tests that the subgraph has the correct nodes."""
    assert [(0, 'node0'), (1, 'node1'), (3, 'node3'), (4, 'node4')] == sorted(self.H.nodes.data('name'))