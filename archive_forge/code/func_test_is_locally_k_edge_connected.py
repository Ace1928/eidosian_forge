import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_is_locally_k_edge_connected():
    G = nx.barbell_graph(10, 0)
    assert is_locally_k_edge_connected(G, 5, 15, k=1)
    assert not is_locally_k_edge_connected(G, 5, 15, k=2)
    G = nx.Graph()
    G.add_nodes_from([5, 15])
    assert not is_locally_k_edge_connected(G, 5, 15, k=2)