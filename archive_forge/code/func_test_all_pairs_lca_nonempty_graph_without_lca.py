from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_nonempty_graph_without_lca(self):
    G = nx.DiGraph()
    G.add_node(3)
    ans = list(all_pairs_lca(G))
    assert ans == [((3, 3), 3)]