from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_default_kwarg(self):
    G = nx.DiGraph([(0, 1), (2, 1)])
    sentinel = object()
    assert nx.lowest_common_ancestor(G, 0, 2, default=sentinel) is sentinel