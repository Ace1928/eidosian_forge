import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_missing_half_edge(self):
    with pytest.raises(nx.NetworkXException):
        embedding = nx.PlanarEmbedding()
        embedding.add_half_edge_first(1, 2)
        embedding.check_structure()