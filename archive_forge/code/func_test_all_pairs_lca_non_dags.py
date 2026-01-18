from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_non_dags(self):
    pytest.raises(nx.NetworkXError, all_pairs_lca, nx.DiGraph([(3, 4), (4, 3)]))