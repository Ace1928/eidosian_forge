import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_navigable_small_world(self):
    G = nx.navigable_small_world_graph(5, p=1, q=0, seed=42)
    gg = nx.grid_2d_graph(5, 5).to_directed()
    assert nx.is_isomorphic(G, gg)
    G = nx.navigable_small_world_graph(5, p=1, q=0, dim=3)
    gg = nx.grid_graph([5, 5, 5]).to_directed()
    assert nx.is_isomorphic(G, gg)
    G = nx.navigable_small_world_graph(5, p=1, q=0, dim=1)
    gg = nx.grid_graph([5]).to_directed()
    assert nx.is_isomorphic(G, gg)