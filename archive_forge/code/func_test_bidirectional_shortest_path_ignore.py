import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_bidirectional_shortest_path_ignore():
    G = nx.Graph()
    nx.add_path(G, [1, 2])
    nx.add_path(G, [1, 3])
    nx.add_path(G, [1, 4])
    pytest.raises(nx.NetworkXNoPath, _bidirectional_shortest_path, G, 1, 2, ignore_nodes=[1])
    pytest.raises(nx.NetworkXNoPath, _bidirectional_shortest_path, G, 1, 2, ignore_nodes=[2])
    G = nx.Graph()
    nx.add_path(G, [1, 3])
    nx.add_path(G, [1, 4])
    nx.add_path(G, [3, 2])
    pytest.raises(nx.NetworkXNoPath, _bidirectional_shortest_path, G, 1, 2, ignore_nodes=[1, 2])