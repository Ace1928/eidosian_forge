from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_identity(self):
    G = nx.DiGraph()
    G.add_node(3)
    assert nx.lowest_common_ancestor(G, 3, 3) == 3