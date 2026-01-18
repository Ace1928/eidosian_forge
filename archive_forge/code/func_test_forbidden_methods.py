import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_forbidden_methods(self):
    embedding = nx.PlanarEmbedding()
    embedding.add_node(42)
    embedding.add_nodes_from([(23, 24)])
    with pytest.raises(NotImplementedError):
        embedding.add_edge(1, 3)
    with pytest.raises(NotImplementedError):
        embedding.add_edges_from([(0, 2), (1, 4)])
    with pytest.raises(NotImplementedError):
        embedding.add_weighted_edges_from([(0, 2, 350), (1, 4, 125)])