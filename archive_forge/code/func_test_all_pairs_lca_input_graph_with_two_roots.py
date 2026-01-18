from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_input_graph_with_two_roots(self):
    G = self.DG.copy()
    G.add_edge(9, 10)
    G.add_edge(9, 4)
    gold = self.gold.copy()
    gold[9, 9] = 9
    gold[9, 10] = 9
    gold[9, 4] = 9
    gold[9, 3] = 9
    gold[10, 4] = 9
    gold[10, 3] = 9
    gold[10, 10] = 10
    testing = dict(all_pairs_lca(G))
    G.add_edge(-1, 9)
    G.add_edge(-1, 0)
    self.assert_lca_dicts_same(testing, gold, G)