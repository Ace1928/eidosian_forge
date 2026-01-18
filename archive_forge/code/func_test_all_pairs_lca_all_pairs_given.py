from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_all_pairs_given(self):
    all_pairs = list(product(self.DG.nodes(), self.DG.nodes()))
    ans = all_pairs_lca(self.DG, pairs=all_pairs)
    self.assert_lca_dicts_same(dict(ans), self.gold)