import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_bidirectional_shortest_path_restricted_wheel():
    wheel = nx.wheel_graph(6)
    length, path = _bidirectional_shortest_path(wheel, 1, 3)
    assert path in [[1, 0, 3], [1, 2, 3]]
    length, path = _bidirectional_shortest_path(wheel, 1, 3, ignore_nodes=[0])
    assert path == [1, 2, 3]
    length, path = _bidirectional_shortest_path(wheel, 1, 3, ignore_nodes=[0, 2])
    assert path == [1, 5, 4, 3]
    length, path = _bidirectional_shortest_path(wheel, 1, 3, ignore_edges=[(1, 0), (5, 0), (2, 3)])
    assert path in [[1, 2, 0, 3], [1, 5, 4, 3]]