import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_paths_source_in_targets():
    G = nx.path_graph(3)
    assert list(nx.all_simple_paths(G, 0, {0, 1, 2})) == [[0], [0, 1], [0, 1, 2]]