from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_null_graph(self):
    pytest.raises(nx.NetworkXPointlessConcept, all_pairs_lca, nx.DiGraph())