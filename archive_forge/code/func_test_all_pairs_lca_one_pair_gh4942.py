from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_one_pair_gh4942(self):
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(2, 0)
    G.add_edge(2, 3)
    G.add_edge(4, 0)
    G.add_edge(5, 2)
    assert nx.lowest_common_ancestor(G, 1, 3) == 2