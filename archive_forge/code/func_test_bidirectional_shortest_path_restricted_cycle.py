import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_bidirectional_shortest_path_restricted_cycle():
    cycle = nx.cycle_graph(7)
    length, path = _bidirectional_shortest_path(cycle, 0, 3)
    assert path == [0, 1, 2, 3]
    length, path = _bidirectional_shortest_path(cycle, 0, 3, ignore_nodes=[1])
    assert path == [0, 6, 5, 4, 3]