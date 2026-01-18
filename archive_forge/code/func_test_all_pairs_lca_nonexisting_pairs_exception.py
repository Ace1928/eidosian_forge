from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_nonexisting_pairs_exception(self):
    pytest.raises(nx.NodeNotFound, all_pairs_lca, self.DG, [(-1, -1)])