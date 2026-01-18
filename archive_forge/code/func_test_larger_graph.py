from itertools import chain, combinations
import pytest
import networkx as nx
def test_larger_graph(self):
    G = nx.gnm_random_graph(100 * self.N, 50 * self.N * self.K)
    nx.community.fast_label_propagation_communities(G)