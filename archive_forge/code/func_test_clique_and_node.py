import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_clique_and_node():
    for n in range(1, 10):
        G = nx.complete_graph(n)
        G.add_node(n + 1)
        _check_augmentations(G, max_k=MAX_EFFICIENT_K + 2)