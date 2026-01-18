from functools import partial
import pytest
import networkx as nx
def test_descendants_at_distance(self):
    for distance, descendants in enumerate([{0}, {1}, {2, 3}, {4}]):
        assert nx.descendants_at_distance(self.G, 0, distance) == descendants