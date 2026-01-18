from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_bug_gh4942(self):
    G = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    ans = list(all_pairs_lca(G))
    assert len(ans) == 9