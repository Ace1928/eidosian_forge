import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_missing_starting_node(self):
    G = nx.path_graph(2)
    assert not nx.is_simple_path(G, [2, 0])