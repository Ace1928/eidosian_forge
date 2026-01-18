import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_invalid_half_edge():
    with pytest.raises(nx.NetworkXException):
        embedding_data = {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}
        embedding = nx.PlanarEmbedding()
        embedding.set_data(embedding_data)
        nx.combinatorial_embedding_to_pos(embedding)