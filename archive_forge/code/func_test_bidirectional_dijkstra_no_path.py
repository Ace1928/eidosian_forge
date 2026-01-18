import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_bidirectional_dijkstra_no_path():
    with pytest.raises(nx.NetworkXNoPath):
        G = nx.Graph()
        nx.add_path(G, [1, 2, 3])
        nx.add_path(G, [4, 5, 6])
        _bidirectional_dijkstra(G, 1, 6)