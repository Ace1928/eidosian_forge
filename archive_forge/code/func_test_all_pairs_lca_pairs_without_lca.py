from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_pairs_without_lca(self):
    G = self.DG.copy()
    G.add_node(-1)
    gen = all_pairs_lca(G, [(-1, -1), (-1, 0)])
    assert dict(gen) == {(-1, -1): -1}