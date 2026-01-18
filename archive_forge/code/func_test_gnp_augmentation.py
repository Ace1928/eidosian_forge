import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def test_gnp_augmentation():
    rng = random.Random(0)
    G = nx.gnp_random_graph(30, 0.005, seed=0)
    avail = {(u, v): 1 + rng.random() for u, v in complement_edges(G) if rng.random() < 0.25}
    _check_augmentations(G, avail)