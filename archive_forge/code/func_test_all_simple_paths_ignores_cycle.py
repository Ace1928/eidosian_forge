import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_paths_ignores_cycle():
    G = nx.cycle_graph(3, create_using=nx.DiGraph())
    G.add_edge(1, 3)
    paths = nx.all_simple_paths(G, 0, 3)
    assert {tuple(p) for p in paths} == {(0, 1, 3)}