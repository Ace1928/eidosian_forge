import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_invalid_edge_orientation(self):
    with pytest.raises(nx.NetworkXException):
        embedding = nx.PlanarEmbedding()
        embedding.add_half_edge_first(1, 2)
        embedding.add_half_edge_first(2, 1)
        embedding.add_edge(1, 3)
        embedding.check_structure()