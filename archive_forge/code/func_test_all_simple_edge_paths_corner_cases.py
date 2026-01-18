import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_edge_paths_corner_cases():
    assert list(nx.all_simple_edge_paths(nx.empty_graph(2), 0, 0)) == []
    assert list(nx.all_simple_edge_paths(nx.empty_graph(2), 0, 1)) == []
    assert list(nx.all_simple_edge_paths(nx.path_graph(9), 0, 8, 0)) == []