import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
def test_dangling_matrix(self):
    """
        Tests that the google_matrix doesn't change except for the dangling
        nodes.
        """
    G = self.G
    dangling = self.dangling_edges
    dangling_sum = sum(dangling.values())
    M1 = nx.google_matrix(G, personalization=dangling)
    M2 = nx.google_matrix(G, personalization=dangling, dangling=dangling)
    for i in range(len(G)):
        for j in range(len(G)):
            if i == self.dangling_node_index and j + 1 in dangling:
                assert M2[i, j] == pytest.approx(dangling[j + 1] / dangling_sum, abs=0.0001)
            else:
                assert M2[i, j] == pytest.approx(M1[i, j], abs=0.0001)