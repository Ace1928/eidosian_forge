import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_get_data(self):
    embedding = self.get_star_embedding(3)
    data = embedding.get_data()
    data_cmp = {0: [2, 1], 1: [0], 2: [0]}
    assert data == data_cmp