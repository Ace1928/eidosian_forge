import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_edge_paths_empty():
    G = nx.path_graph(4)
    paths = nx.all_simple_edge_paths(G, 0, 3, cutoff=2)
    assert list(paths) == []