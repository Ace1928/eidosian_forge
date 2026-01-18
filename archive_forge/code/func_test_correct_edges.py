import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_correct_edges(self):
    """Tests that the subgraph has the correct edges."""
    assert edges_equal([(0, 1, 'edge01'), (3, 4, 'edge34')], self.H.edges.data('name'))