import pytest
import networkx as nx
from networkx.algorithms.planarity import (
@staticmethod
def get_star_embedding(n):
    embedding = nx.PlanarEmbedding()
    for i in range(1, n):
        embedding.add_half_edge_first(0, i)
        embedding.add_half_edge_first(i, 0)
    return embedding