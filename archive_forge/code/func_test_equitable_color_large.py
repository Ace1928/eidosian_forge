import itertools
import pytest
import networkx as nx
def test_equitable_color_large(self):
    G = nx.fast_gnp_random_graph(100, 0.1, seed=42)
    coloring = nx.coloring.equitable_color(G, max_degree(G) + 1)
    assert is_equitable(G, coloring, num_colors=max_degree(G) + 1)