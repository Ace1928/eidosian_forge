import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_is_k_edge_connected():
    G = nx.barbell_graph(10, 0)
    assert is_k_edge_connected(G, k=1)
    assert not is_k_edge_connected(G, k=2)
    G = nx.Graph()
    G.add_nodes_from([5, 15])
    assert not is_k_edge_connected(G, k=1)
    assert not is_k_edge_connected(G, k=2)
    G = nx.complete_graph(5)
    assert is_k_edge_connected(G, k=1)
    assert is_k_edge_connected(G, k=2)
    assert is_k_edge_connected(G, k=3)
    assert is_k_edge_connected(G, k=4)
    G = nx.compose(nx.complete_graph([0, 1, 2]), nx.complete_graph([3, 4, 5]))
    assert not is_k_edge_connected(G, k=1)
    assert not is_k_edge_connected(G, k=2)
    assert not is_k_edge_connected(G, k=3)