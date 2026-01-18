import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_edge_removal(self):
    embedding = nx.PlanarEmbedding()
    embedding.set_data({1: [2, 5, 7], 2: [1, 3, 4, 5], 3: [2, 4], 4: [3, 6, 5, 2], 5: [7, 1, 2, 4], 6: [4, 7], 7: [6, 1, 5]})
    embedding.remove_edges_from(((5, 4), (1, 5)))
    embedding.check_structure()
    embedding_expected = nx.PlanarEmbedding()
    embedding_expected.set_data({1: [2, 7], 2: [1, 3, 4, 5], 3: [2, 4], 4: [3, 6, 2], 5: [7, 2], 6: [4, 7], 7: [6, 1, 5]})
    assert nx.utils.graphs_equal(embedding, embedding_expected)